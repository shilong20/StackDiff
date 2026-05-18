"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf

MPI = None  # type: ignore
_MPI_AVAILABLE = False
_MPI_ENV_KEYS = [
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_RANK",
    "MPI_LOCALNRANKS",
]

if any(os.environ.get(k) for k in _MPI_ENV_KEYS):
    try:
        from mpi4py import MPI  # type: ignore
        _MPI_AVAILABLE = True
    except Exception:
        MPI = None  # type: ignore
        _MPI_AVAILABLE = False
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"
    comm = _get_mpi_comm()
    world_size = comm.Get_size() if comm is not None else 1
    rank = comm.Get_rank() if comm is not None else 0

    if world_size <= 1:
        _setup_single_process(backend)
        return


    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"].strip() == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank % GPUS_PER_NODE}"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    if backend == "nccl" and th.cuda.is_available():
        try:
            num = max(1, th.cuda.device_count())
            local_index = rank % num
            th.cuda.set_device(local_index)
        except Exception:
            pass
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    comm = _get_mpi_comm()
    if comm is None or comm.Get_size() <= 1:
        with bf.BlobFile(path, "rb") as f:
            return th.load(f, **kwargs)

    chunk_size = 2 ** 30
    rank = comm.Get_rank()
    if rank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        comm.bcast(num_chunks, root=0)
        for i in range(0, len(data), chunk_size):
            comm.bcast(data[i : i + chunk_size], root=0)
        buffer = data
    else:
        num_chunks = comm.bcast(None, root=0)
        buffer = bytes()
        for _ in range(num_chunks):
            buffer += comm.bcast(None, root=0)

    return th.load(io.BytesIO(buffer), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if dist.get_world_size() == 1:
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    except OSError:

        return 29500
    finally:
        if s is not None:
            try:
                s.close()
            except OSError:
                pass


def _get_mpi_comm():
    if not _MPI_AVAILABLE:
        return None

    if not any(os.environ.get(k) for k in _MPI_ENV_KEYS):
        return None
    try:
        comm = MPI.COMM_WORLD

        _ = comm.Get_rank()
        _ = comm.Get_size()
        return comm
    except Exception:
        return None


def _setup_single_process(backend: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    # For single GPU, directly install fake distributed to avoid init issues
    _install_fake_distributed()

    # Set CUDA device if available
    if backend == "nccl" and th.cuda.is_available():
        try:
            th.cuda.set_device(0)
        except Exception:
            pass


def _install_fake_distributed() -> None:
    """Internal helper."""

    def _noop(*args, **kwargs):
        return None

    def _return_true():
        return True

    def _return_zero():
        return 0

    def _return_one():
        return 1

    def _fake_broadcast(tensor, src):
        return tensor

    def _fake_all_reduce(tensor, op=None):
        return tensor

    dist.init_process_group = _noop  # type: ignore
    dist.destroy_process_group = _noop  # type: ignore
    dist.is_initialized = _return_true  # type: ignore
    dist.get_rank = _return_zero  # type: ignore
    dist.get_world_size = _return_one  # type: ignore
    dist.barrier = _noop  # type: ignore
    dist.broadcast = _fake_broadcast  # type: ignore
    dist.all_reduce = _fake_all_reduce  # type: ignore
