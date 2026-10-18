#!/usr/bin/env python3
"""Black-box check of hdf5rs's read-write mode (`FileWriter::open_rw`)
using h5py.

Usage (from the repository root):

    cargo build --examples
    python3 tools/check_modified.py [--bin DIR] [--keep DIR]

`--bin` is the directory holding the `modify_demo` and `write_demo`
example binaries (default: target/debug/examples); `--keep` keeps the
files in DIR instead of a temporary directory.

1. h5py creates a base file in several modes (libver earliest / v108 /
   latest, with and without creation-order tracking), `modify_demo
   scenario` modifies it (groups crossing the compact/dense thresholds,
   symbol-table growth, link and attribute deletion, attribute overwrite,
   writes into every chunk index type, resize, append) and h5py checks the
   result.
2. `modify_demo touch` modifies every object of h5py files in each mode;
   h5py checks that everything else is unchanged.
3. Reverse direction: a file written by hdf5rs (`write_demo`) is modified
   by h5py, then by hdf5rs (`touch`), checked by h5py, modified again by
   h5py and read back by hdf5rs (`modify_demo read`).

Exits non-zero on failure. Clean-room reminder: only h5py's public API.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

failures = []
MODES = ["earliest", "v108", "latest", "track", "earliest_track", "fs_aggr"]
# Files hdf5rs must refuse to modify (persistent free-space managers or
# paged allocation).
REFUSED = ["fs_page", "fs_persist"]


def check(cond, what):
    if not cond:
        failures.append(what)
        print(f"FAIL: {what}")


def eq(actual, expected, what):
    try:
        ok = np.array_equal(np.asarray(actual), np.asarray(expected))
    except Exception as e:  # noqa: BLE001
        ok = False
        what = f"{what} ({e})"
    check(ok, f"{what}: got {actual!r}, expected {expected!r}")


def libver(mode):
    if mode.startswith("earliest"):
        return "earliest"
    if mode == "v108":
        return ("v108", "v108")
    return "latest"


def open_new(path, mode):
    h5py.get_config().track_order = mode.endswith("track")
    extra = {}
    if mode == "fs_aggr":
        # Non-default, non-persistent file space strategy: a File Space
        # Info message that hdf5rs must keep valid.
        extra = {"fs_strategy": "aggregate"}
    elif mode == "fs_page":
        extra = {"fs_strategy": "page"}
    elif mode == "fs_persist":
        extra = {"fs_strategy": "fsm", "fs_persist": True}
    return h5py.File(path, "w", libver=libver(mode), track_order=mode.endswith("track"),
                     **extra)


def make_base(path, mode):
    with open_new(path, mode) as f:
        g = f.create_group("grp")
        for i in range(3):
            g.create_group(f"g{i}")
        d = f.create_group("dense_grp")
        for i in range(20):
            d[f"x{i:02}"] = i
        a = f.create_group("attrs")
        for i in range(3):
            a.attrs[f"a{i}"] = np.int64(i)
        da = f.create_group("dense_attrs")
        for i in range(20):
            da.attrs[f"c{i:02}"] = np.int32(i)
        data = f.create_group("data")
        data.create_dataset("fixed", data=np.arange(120, dtype="i4").reshape(10, 12),
                            chunks=(3, 4))
        data.create_dataset("ea", data=np.arange(25, dtype="f8"), chunks=(4,),
                            maxshape=(None,))
        data.create_dataset("bt2", data=np.arange(42, dtype="i4").reshape(6, 7),
                            chunks=(2, 3), maxshape=(None, None))
        data.create_dataset("gz", data=np.arange(100, dtype="i4"), chunks=(10,),
                            compression="gzip", shuffle=True)
        data.create_dataset("contig", data=np.arange(20, dtype="f4"))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_layout(h5py.h5d.COMPACT)
        sid = h5py.h5s.create_simple((5,))
        h5py.h5d.create(data.id, b"compact", h5py.h5t.STD_I16LE, sid, dcpl=dcpl)
        data["compact"][...] = np.arange(5, dtype="i2")
        data.create_dataset("single", data=np.arange(20, dtype="i4").reshape(4, 5),
                            chunks=(4, 5))
        f["to_delete"] = np.arange(3)
        f["hard_a"] = np.arange(4)
        f["hard_b"] = f["hard_a"]
    h5py.get_config().track_order = False


def check_scenario(path, mode):
    tracked = mode.endswith("track")
    with h5py.File(path, "r") as f:
        eq(f.attrs["modified_by"].decode() if isinstance(f.attrs["modified_by"], bytes)
           else f.attrs["modified_by"], "hdf5rs", f"{mode}: /@modified_by")
        keys = list(f["grp"].keys())
        new = [f"rw_g{i}" for i in range(10)]
        if tracked:
            eq(keys, ["g0", "g1", "g2"] + new, f"{mode}: /grp order")
        else:
            eq(sorted(keys), sorted(["g0", "g1", "g2"] + new), f"{mode}: /grp members")
        eq(sorted(f["dense_grp"].keys()), [f"x{i:02}" for i in range(15, 20)],
           f"{mode}: /dense_grp members")
        for i in range(15, 20):
            eq(f[f"dense_grp/x{i:02}"][()], i, f"{mode}: /dense_grp/x{i}")
        a = f["attrs"].attrs
        names = list(a.keys())
        expect = ["a0", "a2"] + [f"rw_b{i:02}" for i in range(15)]
        if tracked:
            # a0 was rewritten: it is now the newest of the original ones.
            check(names[-15:] == expect[2:], f"{mode}: /attrs order {names}")
        eq(sorted(names), sorted(expect), f"{mode}: /attrs names")
        eq(a["a0"], 100, f"{mode}: /attrs@a0")
        eq(a["a2"], 2, f"{mode}: /attrs@a2")
        for i in range(15):
            eq(a[f"rw_b{i:02}"], i, f"{mode}: /attrs@rw_b{i}")
        da = f["dense_attrs"].attrs
        eq(sorted(da.keys()), ["c18", "c19"], f"{mode}: /dense_attrs names")
        eq(da["c19"], 19, f"{mode}: /dense_attrs@c19")
        fixed = np.arange(120, dtype="i4").reshape(10, 12)
        fixed[1:3, 2:4] = 99
        eq(f["data/fixed"][()], fixed, f"{mode}: fixed")
        eq(f["data/ea"][()], np.concatenate([np.arange(25.0), 1000.0 + np.arange(10)]),
           f"{mode}: ea")
        bt2 = np.zeros((8, 9), dtype="i4")
        bt2[:6, :7] = np.arange(42).reshape(6, 7)
        bt2[6:8, :] = -1
        eq(f["data/bt2"][()], bt2, f"{mode}: bt2")
        gz = np.arange(100, dtype="i4")
        gz[5:25] = -5
        eq(f["data/gz"][()], gz, f"{mode}: gz")
        c = np.arange(20, dtype="f4")
        c[3] = 3.5
        eq(f["data/contig"][()], c, f"{mode}: contig")
        eq(f["data/compact"][()], [7, 8, 2, 3, 4], f"{mode}: compact")
        s = np.arange(20, dtype="i4").reshape(4, 5)
        s[0, 0] = 42
        eq(f["data/single"][()], s, f"{mode}: single")
        check("to_delete" not in f, f"{mode}: /to_delete deleted")
        check("hard_b" not in f, f"{mode}: /hard_b deleted")
        eq(f["hard_a"][()], np.arange(4), f"{mode}: /hard_a kept")
        eq(f["new/ds"][()], [1, 2, 3], f"{mode}: /new/ds")
        by = f["new/ds"].attrs["by"]
        eq(by.decode() if isinstance(by, bytes) else by, "hdf5rs", f"{mode}: /new/ds@by")
    # The reference library can modify the result further.
    with h5py.File(path, "r+") as f:
        f["grp"].create_group("after")
        del f["grp/rw_g3"]
        f["data/ea"].resize((40,))
        f["data/ea"][35:] = -1
        f["attrs"].attrs["late"] = 1
        del f["attrs"].attrs["rw_b00"]
        f["dense_grp"]["y"] = 5
    with h5py.File(path, "r") as f:
        check("after" in f["grp"] and "rw_g3" not in f["grp"], f"{mode}: h5py relink")
        eq(f["data/ea"][35:], [-1] * 5, f"{mode}: h5py resize")
        check("late" in f["attrs"].attrs and "rw_b00" not in f["attrs"].attrs,
              f"{mode}: h5py attrs")


def value_repr(v):
    a = np.asarray(v)
    if a.dtype.kind == "O":
        return repr([x.tobytes() if isinstance(x, np.ndarray) else x for x in a.ravel()])
    if a.dtype.kind == "V" and a.dtype.names is None:
        return a.tobytes().hex()
    try:
        return a.dtype.str + str(a.shape) + a.tobytes().hex()
    except Exception:  # noqa: BLE001
        return repr(v)


def read_dataset(d):
    try:
        return value_repr(d[()])
    except Exception as e:  # noqa: BLE001
        try:
            buf = np.empty(d.shape, dtype=f"V{d.dtype.itemsize}")
            d.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, mtype=d.id.get_type())
            return buf.tobytes().hex()
        except Exception:  # noqa: BLE001
            return f"error {type(e).__name__}"


def dump(f):
    """Everything reachable, keyed by path, without names starting rw_."""
    out = {}
    seen = {}

    def attrs(obj):
        r = []
        for k in sorted(obj.attrs.keys()):
            if k.startswith("rw_"):
                continue
            try:
                r.append((k, value_repr(obj.attrs[k])))
            except Exception as e:  # noqa: BLE001
                r.append((k, f"error {type(e).__name__}"))
        return r

    def rec(path, obj):
        key = hash(obj.id)  # identifies the object (file and header)
        if key in seen:
            out[path] = f"alias of {seen[key]}"
            return
        seen[key] = path
        if isinstance(obj, h5py.Group):
            links = []
            for k in sorted(obj.keys()):
                if k.startswith("rw_"):
                    continue
                link = obj.get(k, getlink=True)
                if isinstance(link, h5py.SoftLink):
                    links.append((k, "soft", link.path))
                elif isinstance(link, h5py.ExternalLink):
                    links.append((k, "ext", link.filename, link.path))
                else:
                    links.append((k, "hard"))
            out[path] = ("group", links, attrs(obj))
            for k, kind, *_ in links:
                if kind == "hard":
                    rec(f"{path.rstrip('/')}/{k}", obj[k])
        elif isinstance(obj, h5py.Dataset):
            out[path] = ("dataset", str(obj.dtype), obj.shape, obj.maxshape,
                         read_dataset(obj), attrs(obj))
        else:
            out[path] = ("datatype", str(obj.dtype), attrs(obj))

    rec("/", f["/"])
    return out


def check_touched(path, before, what):
    with h5py.File(path, "r") as f:
        after = dump(f)
        for k, v in before.items():
            check(after.get(k) == v, f"{what}: {k} changed: {after.get(k)!r} != {v!r}")
        for k in after:
            check(k in before, f"{what}: unexpected {k}")
        eq(f.attrs["rw_attr"], 42, f"{what}: /@rw_attr")
        eq(f["rw_group/d"][()], [7, 8, 9], f"{what}: /rw_group/d")
        n = [0]

        def visit(name, obj):
            if name.startswith("rw_group"):
                return
            n[0] += 1
            check("rw_a" in obj.attrs, f"{what}: {name}@rw_a")
            if isinstance(obj, h5py.Group):
                eq(obj["rw_soft"][()], [7, 8, 9], f"{what}: {name}/rw_soft")
        f.visititems(visit)
        print(f"{what}: {n[0]} objects checked")


def run(bindir, *args):
    exe = os.path.join(bindir, args[0])
    r = subprocess.run([exe, *args[1:]], capture_output=True, text=True)
    if r.returncode != 0:
        check(False, f"{' '.join(args)} failed: {r.stderr.strip()}")
        return False
    print(r.stdout.strip())
    return True


def make_rich(path, mode):
    """A file with a bit of everything, for the generic modifications."""
    with open_new(path, mode) as f:
        f.attrs["title"] = "rich"
        f.attrs["vlen"] = "variable length"
        g = f.create_group("a/b/c")
        g["leaf"] = np.arange(6).reshape(2, 3)
        m = f.create_group("many")
        for i in range(40):
            m[f"d{i:02}"] = np.float32(i)
        for i in range(12):
            m.attrs[f"m{i:02}"] = i
        f.create_dataset("strings", data=["a", "bb", "ccc"], dtype=h5py.string_dtype())
        f.create_dataset("chunked", data=np.arange(1000), chunks=(64,), maxshape=(None,),
                         compression="gzip")
        f.create_dataset("chunk2d", data=np.arange(400.0).reshape(20, 20), chunks=(7, 6))
        f.create_dataset("lzf", data=np.arange(500, dtype="i2"), chunks=(50,),
                         compression="lzf")
        f.create_dataset("fletcher", data=np.arange(50), chunks=(10,), fletcher32=True)
        f.create_dataset("empty", shape=(10,), dtype="i4", fillvalue=-3)
        f["dt"] = np.dtype([("x", "f8"), ("y", "i4")])
        f.create_dataset("typed", data=np.zeros(3, dtype=f["dt"]))
        f["soft"] = h5py.SoftLink("/a/b/c/leaf")
        f["alias"] = f["a/b/c/leaf"]
        f["refs"] = np.array([f["chunked"].ref, f["a"].ref], dtype=h5py.ref_dtype)
    h5py.get_config().track_order = False


def h5py_modify_demo_file(path):
    """Modify a file written by hdf5rs's write_demo with h5py."""
    with h5py.File(path, "r+") as f:
        f.create_group("by_h5py").attrs["x"] = 1
        f["by_h5py"].create_dataset("data", data=np.arange(30), chunks=(8,),
                                    maxshape=(None,))
        f["numbers"].attrs["h5py"] = "yes"
        for k in list(f["many"].keys())[:3]:
            del f["many"][k]


def main():
    args = sys.argv[1:]
    bindir = "target/debug/examples"
    keep = None
    while args:
        a = args.pop(0)
        if a == "--bin":
            bindir = args.pop(0)
        elif a == "--keep":
            keep = args.pop(0)
        else:
            print(__doc__)
            return 2
    tmp = keep or tempfile.mkdtemp(prefix="hdf5rs-modify-")
    os.makedirs(tmp, exist_ok=True)
    print(f"h5py {h5py.version.version}, libhdf5 {h5py.version.hdf5_version}; files in {tmp}")
    for mode in MODES:
        p = os.path.join(tmp, f"scenario-{mode}.h5")
        make_base(p, mode)
        if run(bindir, "modify_demo", "scenario", p):
            try:
                check_scenario(p, mode)
            except Exception as e:  # noqa: BLE001
                check(False, f"{mode}: scenario check raised {type(e).__name__}: {e}")
            run(bindir, "modify_demo", "read", p)
        for kind, maker in [("base", make_base), ("rich", make_rich)]:
            p = os.path.join(tmp, f"touch-{kind}-{mode}.h5")
            maker(p, mode)
            with h5py.File(p, "r") as f:
                before = dump(f)
            if run(bindir, "modify_demo", "touch", p):
                try:
                    check_touched(p, before, f"touch {kind} {mode}")
                except Exception as e:  # noqa: BLE001
                    check(False, f"touch {kind} {mode} raised {type(e).__name__}: {e}")
    for mode in REFUSED:
        p = os.path.join(tmp, f"refused-{mode}.h5")
        make_base(p, mode)
        with open(p, "rb") as fh:
            orig = fh.read()
        r = subprocess.run([os.path.join(bindir, "modify_demo"), "touch", p],
                           capture_output=True, text=True)
        check(r.returncode != 0 and "unsupported" in r.stderr.lower(),
              f"{mode}: expected a refusal, got {r.returncode} {r.stderr.strip()}")
        with open(p, "rb") as fh:
            check(fh.read() == orig, f"{mode}: refused file was changed")
        print(f"{mode}: refused ({r.stderr.strip()})")
    # Reverse: hdf5rs file -> h5py -> hdf5rs -> h5py -> hdf5rs.
    for v in ["v0", "v2", "v3"]:
        p = os.path.join(tmp, f"demo-{v}.h5")
        if not run(bindir, "write_demo", p, v):
            continue
        try:
            h5py_modify_demo_file(p)
            with h5py.File(p, "r") as f:
                before = dump(f)
        except Exception as e:  # noqa: BLE001
            check(False, f"demo {v}: h5py modification raised {type(e).__name__}: {e}")
            continue
        if run(bindir, "modify_demo", "touch", p):
            try:
                check_touched(p, before, f"demo {v}")
                with h5py.File(p, "r+") as f:
                    f["rw_group"].create_dataset("from_h5py", data=np.arange(3))
                    del f["rw_group/d"]
                    f["by_h5py/data"].resize((40,))
            except Exception as e:  # noqa: BLE001
                check(False, f"demo {v}: check raised {type(e).__name__}: {e}")
            run(bindir, "modify_demo", "read", p)
    if not keep:
        shutil.rmtree(tmp)
    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
