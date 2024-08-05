import sys
import io
import os
import re
import json
import tarfile
from functools import partial

import webdataset as wds
from webdataset import ResampledShards, DataPipeline, tarfile_to_samples
from webdataset.filters import pipelinefilter
from webdataset.tariterators import url_opener, group_by_keys
from webdataset.handlers import reraise_exception
from webdataset.gopen import gopen_schemes, gopen


def pytorch_worker_info(group=None):  # sourcery skip: use-contextlib-suppress
    """Return node and worker info for PyTorch and some distributed environments."""
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
    try:
        import torch.distributed

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            group = group or torch.distributed.group.WORLD
            rank = torch.distributed.get_rank(group=group)
            world_size = torch.distributed.get_world_size(group=group)
    except ModuleNotFoundError:
        pass
    try:
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker = worker_info.id
            num_workers = worker_info.num_workers
    except ModuleNotFoundError:
        pass

    return rank, world_size, worker, num_workers


def pytorch_worker_seed(group=None):
    """Compute a distinct, deterministic RNG seed for each worker and node."""
    rank, world_size, worker, num_workers = pytorch_worker_info(group=group)
    return rank * 1000 + worker


def worker_seed_sat(group=None, seed=0):
    return pytorch_worker_seed(group=group) + seed * 23


class ConfiguredResampledShards(ResampledShards):
    def __init__(self, urls, seed, nshards=sys.maxsize, deterministic=True):
        from sat.helpers import print_rank0

        try:
            from megatron.core.parallel_state import get_data_parallel_group

            group = get_data_parallel_group()
            print_rank0("Using megatron data parallel group.")
        except:
            from sat.mpu import get_data_parallel_group

            try:
                group = get_data_parallel_group()
                print_rank0("Using sat data parallel group.")
            except AssertionError:
                group = None
                print_rank0("No data parallel group is specified!")
        worker_seed_sat_this = partial(worker_seed_sat, group=group, seed=seed)
        super().__init__(urls, nshards, worker_seed_sat_this, deterministic)


class SimpleDistributedWebDataset(DataPipeline):
    def __init__(self, path, process_fn, seed, *, shuffle_buffer=1000):
        # set shuffle_buffer = 1 to disable it, model-parallel will be different due to shuffle
        try:
            from sat.mpu import get_model_parallel_world_size

            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1
        except Exception:
            pass
        super().__init__(
            ConfiguredResampledShards(path, seed),  # Lots of shards are recommended, or not evenly
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            process_fn,
        )


def tar_file_iterator_with_meta(
    fileobj, meta_names, skip_meta=r"__[^/]*__($|/)", suffix=None, handler=reraise_exception, meta_stream=None
):
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    :param fileobj: byte stream suitable for tarfile
    :param meta_names: key of different items in meta file
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")

    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    data_dir, filename = fileobj.name.rsplit("/", 1)
    meta_data = {}  # {id: {meta_name: meta_value, meta_name2: meta_value2, ...}}

    if meta_stream is None:
        meta_file_name = filename.split(".")[0] + ".meta.jsonl"
        meta_path = os.path.join(data_dir, meta_file_name)
        if os.path.exists(meta_path):
            meta_stream = open(meta_path, "r")
    else:
        meta_file_name = meta_stream.name

    if meta_stream is not None:
        for lineno, line in enumerate(meta_stream):
            meta_list = []
            try:
                meta_list.append(json.loads(line))
            except Exception as exn:
                from sat.helpers import print_rank0

                print_rank0(f"Error in loading jsonl {meta_file_name}, lineno {lineno}: {line}", level="DEBUG")
                continue
            for item in meta_list:
                if not item["key"] in meta_data:
                    meta_data[item["key"]] = {}
                for meta_name in meta_names:
                    if meta_name in item:
                        meta_data[item["key"]][meta_name] = item[meta_name]
        meta_stream.close()

    try:
        for tarinfo in stream:
            fname = tarinfo.name
            try:
                if not tarinfo.isreg():
                    continue
                if fname is None:
                    continue
                if "/" not in fname and fname.startswith("__") and fname.endswith("__"):
                    # skipping metadata for now
                    continue
                if skip_meta is not None and re.match(skip_meta, fname):
                    continue
                if fname.endswith(".txt") and suffix is not None:
                    data = (stream.extractfile(tarinfo).read().decode() + suffix).encode()
                else:
                    data = stream.extractfile(tarinfo).read()
                result = dict(fname=fname, data=data)
                yield result

                if fname.endswith(".id"):
                    fid = fname.split(".")[0]
                    if "-$#%@&" in fid:
                        sfid = fid.split("-$#%@&")[0]
                    else:
                        sfid = fid
                    meta_data_fid = meta_data.get(sfid, {})
                    for meta_name in meta_names:
                        meta_fname = fid + "." + meta_name
                        meta = meta_data_fid.get(meta_name, None)
                        yield dict(fname=meta_fname, data=meta)
                stream.members = []
            except Exception as exn:
                if hasattr(exn, "args") and len(exn.args) > 0:
                    exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
                if handler(exn):
                    continue
                else:
                    break
    except Exception as exn:
        print(exn)
    del stream


def tar_file_expander_with_meta(data, meta_names, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator_with_meta(source["stream"], meta_names, meta_stream=source["meta_stream"]):
                assert isinstance(sample, dict) and "data" in sample and "fname" in sample
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def url_opener(
    data,
    handler,
    **kw,
):
    """Open URLs and yield a stream of url+stream pairs.

    Args:
        data: iterator over dict(url=...)
        handler: exception handler.
        kw: keyword arguments for gopen.gopen.

    Yields:
        a stream of url+stream pairs.
    """
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            stream = gopen(url, **kw)
            if hasattr(stream, "meta_stream"):
                meta_stream = stream.meta_stream
                del stream.meta_stream
            else:
                meta_stream = None
            sample.update(stream=stream, meta_stream=meta_stream)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


def tarfile_samples_with_meta(src, meta_names, handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander_with_meta(streams, meta_names, handler)
    samples = group_by_keys(files, handler=handler)
    return samples


class MetaDistributedWebDataset(DataPipeline):
    """WebDataset with meta information files
    Extra Format:
        in webdataset (tar), for each sample there is a '.id';
        for each tar file, there is a '.meta.jsonl' file with the same name;
        The '.meta.jsonl' file contains lines of json objects, each with a 'key' field to match '.id'.
    """

    def __init__(
        self, path, process_fn, seed, *, meta_names=[], nshards=sys.maxsize, shuffle_buffer=1000, include_dirs=None
    ):
        # os.environ['WDS_SHOW_SEED'] = '1'
        import torch

        if torch.distributed.get_rank() == 0:
            if include_dirs is not None:  # /webdatasets/A,/webdatasets/C
                other_paths = []
                include_dirs = include_dirs.split(",")
                for include_dir in include_dirs:
                    if "*" in include_dir:
                        include_dir, n = include_dir.split("*")
                        n = int(n)
                    else:
                        n = 1
                    for cur_dir, dirs, files in os.walk(include_dir):
                        for f in files:
                            if f.endswith("tar") and os.path.getsize(os.path.join(cur_dir, f)) > 0:
                                # other_paths.append(os.path.join(cur_dir,f))
                                other_paths.extend([os.path.join(cur_dir, f)] * n)
                # print(f'Adding dataset paths {",".join(other_paths)}')
                from braceexpand import braceexpand

                if len(path) > 0:  # not ""
                    path = list(braceexpand(path)) + other_paths
                else:
                    path = other_paths
            path = [path]
        else:
            path = [
                None,
            ]
        torch.distributed.broadcast_object_list(path, src=0)
        path = path[0]

        tarfile_samples = partial(tarfile_samples_with_meta, meta_names=meta_names)
        tarfile_to_samples = pipelinefilter(tarfile_samples)

        # if model parallel, shuffle_buffer should be 1 to disable shuffling
        try:
            from sat.mpu import get_model_parallel_world_size

            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1
        except Exception:
            pass

        super().__init__(
            ConfiguredResampledShards(path, seed, nshards=nshards),
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            process_fn,
        )


# rclone support
from webdataset.gopen import Pipe


def gopen_rclone(url, mode="rb", bufsize=1024 * 1024 * 32):
    """Open a URL with `curl`.

    :param url: rclone url, e.g. data:bucket1/foo.tar. data should be configured.
    :param mode: file mode
    :param bufsize: buffer size
    """
    url = url.replace("rclone://", "")
    if mode[0] == "r":
        cmd = f"rclone cat '{url}'"
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 23],
        )  # skipcq: BAN-B604
    elif mode[0] == "w":
        cmd = f"rclone cp - '{url}'"
        return Pipe(
            cmd,
            mode=mode,
            shell=True,
            bufsize=bufsize,
            ignore_status=[141, 26],
        )  # skipcq: BAN-B604
    else:
        raise ValueError(f"{mode}: unknown mode")


def gopen_boto3(url, mode="rb", bufsize=8192 * 2):
    """Open a URL with boto3 API.

    :param url: boto3 url, e.g. boto3://bucket1/foo.tar. data should be configured.
    :param mode: file mode
    :param bufsize: buffer size
    """
    import boto3

    # boto3.set_stream_logger('botocore', level='DEBUG')
    if url.startswith("boto3://"):
        url = url.replace("boto3://", "")
        need_meta = False
    else:
        url = url.replace("metaboto3://", "")
        need_meta = True
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", None)
    access_key = os.environ.get("S3_ACCESS_KEY_ID", None)
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY", None)

    if mode[0] == "r":
        s3_client = boto3.client(
            "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
        bucket, key = url.split("/", 1)

        if need_meta:
            # download a meta json
            meta_file_key = key.split(".")[0] + ".meta.jsonl"
            meta_stream = io.BytesIO()
            s3_client.download_fileobj(bucket, meta_file_key, meta_stream)
            meta_stream.seek(0)
            meta_stream.name = meta_file_key
        else:
            meta_stream = None

        # data tar stream
        response = s3_client.get_object(Bucket=bucket, Key=key)  # Range optional
        response["Body"].name = key  # actually not used
        response["Body"].meta_stream = meta_stream
        return response["Body"]
    else:
        raise ValueError(f"{mode}: unknown mode")


gopen_schemes["rclone"] = gopen_rclone
gopen_schemes["boto3"] = gopen_boto3
gopen_schemes["metaboto3"] = gopen_boto3
