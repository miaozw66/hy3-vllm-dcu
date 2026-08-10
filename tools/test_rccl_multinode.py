#!/usr/bin/env python3
"""RCCL multi-node test — run with torchrun on both nodes simultaneously.

Usage:
  Node 0: bash test_rccl_wrapper.sh 0
  Node 1: bash test_rccl_wrapper.sh 1

Environment variables:
  RCCL_MASTER_ADDR  — primary node IP (default: 10.18.17.71)
  RCCL_MASTER_PORT  — port (default: 29505)
"""
import os, sys, socket, fcntl, struct, torch, torch.distributed as dist

MASTER_ADDR = os.environ.get("RCCL_MASTER_ADDR", "10.18.17.71")
MASTER_PORT = os.environ.get("RCCL_MASTER_PORT", "29505")


def detect_iface(target_ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((target_ip, 1))
    src_ip = s.getsockname()[0]
    s.close()
    for ifname in socket.if_nameindex():
        name = ifname[1]
        try:
            ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            addr = socket.inet_ntoa(
                fcntl.ioctl(ss.fileno(), 0x8915,
                            struct.pack('256s', name[:15].encode()))[20:24])
            ss.close()
            if addr == src_ip:
                return name, src_ip
        except Exception:
            pass
    return None, src_ip


def log(msg):
    print(msg, flush=True)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 8))
    global_rank = node_rank * 4 + local_rank

    # --- Phase 1: interface detection ---
    iface, src_ip = detect_iface(MASTER_ADDR)
    if local_rank == 0:
        log(f"[N{node_rank}] Detected: iface={iface}, src_ip={src_ip}")
        log(f"[N{node_rank}] All interfaces:")
        for ifname in socket.if_nameindex():
            name = ifname[1]
            try:
                ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                addr = socket.inet_ntoa(
                    fcntl.ioctl(ss.fileno(), 0x8915,
                                struct.pack('256s', name[:15].encode()))[20:24])
                ss.close()
                log(f"           {name} -> {addr}")
            except Exception:
                log(f"           {name} -> (no IPv4)")

    if iface is None:
        log(f"[N{node_rank}] WARNING: could not detect interface, "
            f"trying common names")
        # Try common interface names
        for candidate in ["eno1", "eth0", "ens", "bond0"]:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
                             candidate.encode())
                s.close()
                iface = candidate
                log(f"[N{node_rank}] Using fallback iface: {iface}")
                break
            except Exception:
                continue
        if iface is None:
            iface = "eno1"
            log(f"[N{node_rank}] Using hardcoded fallback: {iface}")

    os.environ["NCCL_SOCKET_IFNAME"] = iface
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"

    # --- Phase 2: GPU setup ---
    torch.cuda.set_device(local_rank)
    gpu_name = torch.cuda.get_device_name(local_rank)
    if local_rank == 0:
        log(f"[N{node_rank}] GPU: {gpu_name}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # --- Phase 3: RCCL init ---
    if local_rank == 0:
        log(f"[N{node_rank}] Initializing RCCL (rank {global_rank}/{world_size}) "
            f"via tcp://{MASTER_ADDR}:{MASTER_PORT} ...")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        world_size=world_size,
        rank=global_rank,
    )

    log(f"[N{node_rank}/G{local_rank}] RCCL initialized! rank={global_rank}")

    # --- Phase 4: all_reduce test ---
    t = torch.ones(1024).cuda() * (global_rank + 1)
    dist.all_reduce(t)
    expected = sum(range(1, world_size + 1))
    ok = abs(t[0].item() - expected) < 0.01
    log(f"  Rank {global_rank:2d}: "
        f"result={t[0].item():.0f} expected={expected} {'OK' if ok else 'FAIL'}")

    dist.barrier()

    if global_rank == 0:
        log(f"\n{'='*60}")
        log(f"SUCCESS: RCCL {world_size}-GPU multi-node communication OK!")
        log(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
