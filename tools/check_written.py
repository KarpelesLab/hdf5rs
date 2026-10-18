#!/usr/bin/env python3
"""Black-box check of files produced by hdf5rs's writer, using h5py.

Usage (from the repository root):

    cargo run --example write_demo -- /tmp/demo.h5 [v2|v3]
    python3 tools/check_written.py /tmp/demo.h5

Opens the file with the reference library (through h5py's public API only)
and asserts the structure and values written by `examples/write_demo.rs`.
Also visits every object and reads every attribute and dataset so that
anything libhdf5 cannot decode is reported. Exits non-zero on failure.
"""

import sys

import h5py
import numpy as np

failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print(f"FAIL: {what}")


def eq(actual, expected, what):
    try:
        ok = np.array_equal(np.asarray(actual), np.asarray(expected))
    except Exception as e:  # noqa: BLE001 - report any comparison problem
        ok = False
        what = f"{what} ({e})"
    check(ok, f"{what}: got {actual!r}, expected {expected!r}")


def s(v):
    return v.decode() if isinstance(v, bytes) else v


def read_opaque(d):
    """Read a tagged opaque dataset (h5py has no conversion path to a
    NumPy void type for arbitrary tags, so read with the file type)."""
    buf = np.empty(d.shape, dtype=f"V{d.dtype.itemsize}")
    d.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, mtype=d.id.get_type())
    return buf


def check_root(f):
    a = f.attrs
    eq(s(a["title"]), "hdf5rs write demo", "/@title")
    eq(a["version"], 3, "/@version")
    check(a["version"].dtype == np.int32, "/@version dtype")
    eq(a["scale"], [0.5, 1.5, 2.5], "/@scale")
    eq(a["flag"], True, "/@flag")
    eq([s(x) for x in a["names"]], ["alpha", "beta", "γ"], "/@names")
    eq(a["fixed"], b"abc", "/@fixed")
    fdt = a.get_id("fixed").dtype
    check(fdt == np.dtype("S8"), f"/@fixed dtype {fdt}")
    check(
        sorted(f.keys()) == ["a", "chunked", "links", "many", "numbers", "strings",
                         "types"],
        f"root members {sorted(f.keys())}",
    )


def check_numbers(f):
    n = f["numbers"]
    for name, dt, vals in [
        ("i8", "i1", [-128, -1, 0, 1, 127]),
        ("u8", "u1", [0, 1, 255]),
        ("i16", "<i2", [-32768, 0, 32767]),
        ("u16", "<u2", [0, 65535]),
        ("i32", "<i4", [-(2**31), 0, 2**31 - 1]),
        ("u32", "<u4", [0, 2**32 - 1]),
        ("i64", "<i8", [-(2**63), 0, 2**63 - 1]),
        ("u64", "<u8", [0, 2**64 - 1]),
        ("f32", "<f4", [1.5, -0.25, np.inf]),
    ]:
        d = n[name]
        check(d.dtype == np.dtype(dt), f"numbers/{name} dtype {d.dtype} != {dt}")
        eq(d[()], np.array(vals, dtype=dt), f"numbers/{name}")
    g = n["grid"]
    check(g.shape == (3, 4), "grid shape")
    eq(g[()], (np.arange(12) * 0.5).reshape(3, 4), "grid values")
    eq(s(g.attrs["units"]), "m", "grid@units")
    eq(g.attrs["valid_range"], [0.0, 10.0], "grid@valid_range")
    check(n["scalar"].shape == (), "scalar shape")
    eq(n["scalar"][()], 42.0, "scalar value")
    check(n["bools"].dtype == np.bool_, f"bools dtype {n['bools'].dtype}")
    eq(n["bools"][()], [True, False, True], "bools")
    c = n["compact"]
    check(c.id.get_create_plist().get_layout() == h5py.h5d.COMPACT, "compact layout")
    eq(c[()], [1, 2, 3, 4], "compact values")
    fl = n["filled"]
    check(fl.shape == (2, 5), "filled shape")
    eq(fl.fillvalue, -7, "filled fillvalue")
    eq(fl[()], np.full((2, 5), -7, dtype="i4"), "filled values")
    check(n["empty"].shape == (0,), "empty shape")
    eq(n["empty"][()], np.zeros(0, dtype="f4"), "empty values")
    be = n["be_u32"]
    check(be.dtype == np.dtype(">u4"), f"be_u32 dtype {be.dtype}")
    eq(be[()], [256, 2], "be_u32 values")


def check_strings(f):
    st = f["strings"]
    v = st["vlen"]
    check(h5py.check_string_dtype(v.dtype) is not None, "vlen is a string dtype")
    eq([s(x) for x in v[()]], ["hello", "", "wörld"], "strings/vlen")
    eq(s(st["vlen_scalar"][()]), "just one", "strings/vlen_scalar")
    fx = st["fixed"]
    check(fx.dtype == np.dtype("S6"), f"fixed dtype {fx.dtype}")
    eq(fx[()], [b"ab", b"cdef", b"ghijkl"], "strings/fixed")
    fu = st["fixed_utf8"]
    info = h5py.check_string_dtype(fu.dtype)
    check(info is not None and info.encoding == "utf-8" and info.length == 8,
          f"fixed_utf8 dtype info {info}")
    eq(fu[0].decode("utf-8"), "café", "strings/fixed_utf8")


def check_types(f):
    t = f["types"]
    p = t["points"][()]
    check(p.dtype.names == ("x", "y", "id"), f"points fields {p.dtype.names}")
    eq(p["x"], [1.0, -1.0], "points.x")
    eq(p["y"], [2.0, 0.5], "points.y")
    eq(p["id"], [7, 8], "points.id")
    col = t["colors"]
    mapping = h5py.check_enum_dtype(col.dtype)
    check(mapping == {"RED": 0, "GREEN": 1, "BLUE": 2}, f"colors enum {mapping}")
    eq(col[()], [0, 2, 1, 1], "colors values")
    arr = t["arrays"]
    check(arr.dtype.subdtype is not None or arr[()].shape == (2, 3),
          f"arrays dtype {arr.dtype}")
    eq(arr[()], [[1, 2, 3], [4, 5, 6]], "arrays values")
    rag = t["ragged"][()]
    check([[int(v) for v in x] for x in rag] == [[1], [], [2, 3, 4]],
          f"ragged values {rag!r}")
    op = t["opaque"]
    check(op.dtype.kind == "V" and op.dtype.itemsize == 4, f"opaque dtype {op.dtype}")
    check(op.id.get_type().get_tag() == b"demo-tag", "opaque tag")
    eq(read_opaque(op).tobytes(), bytes(range(1, 9)), "opaque bytes")
    bits = t["bits"]
    eq(bits[()], [0b1010_0101, 0xFF], "bitfield values")
    pad = t["padded"]
    check(pad.dtype.itemsize == 16 and pad.dtype.fields["b"][1] == 8,
          f"padded dtype {pad.dtype}")
    eq(pad[()]["a"], [9], "padded.a")
    eq(pad[()]["b"], [2.25], "padded.b")
    nt = t["point_t"]
    check(isinstance(nt, h5py.Datatype), "point_t is a committed datatype")
    npnt = t["named_points"]
    check(npnt.id.get_type().committed(), "named_points uses a committed type")
    eq(npnt[()]["x"], [3.0], "named_points.x")
    eq(npnt[()]["id"], [5], "named_points.id")
    check(nt.id.committed(), "point_t committed")


def is_compat(f):
    return f.id.get_create_plist().get_version()[0] == 0


def check_links(f):
    lk = f["links"]
    check(lk.get("grid_again", getlink=True).__class__ is h5py.HardLink, "grid_again hard")
    check(lk["grid_again"] == f["numbers/grid"], "grid_again same object")
    sl = lk.get("soft_grid", getlink=True)
    check(isinstance(sl, h5py.SoftLink) and sl.path == "/numbers/grid", "soft_grid")
    eq(lk["soft_grid"][()], f["numbers/grid"][()], "soft_grid deref")
    dl = lk.get("dangling", getlink=True)
    check(isinstance(dl, h5py.SoftLink) and dl.path == "/does/not/exist", "dangling")
    if is_compat(f):
        check("ext" not in lk, "no external link in compatibility format")
    else:
        el = lk.get("ext", getlink=True)
        check(isinstance(el, h5py.ExternalLink) and el.filename == "other.h5"
              and el.path == "/some/object", f"external link {el}")
    if f.id.get_create_plist().get_sizes()[0] == 8:
        refs = lk["refs"][()]
        check(f[refs[0]] == f["numbers/grid"], "refs[0] -> grid")
        check(f[refs[1]] == f["strings"], "refs[1] -> strings")
        check(not refs[2], "refs[2] is null")
        check(f[lk.attrs["grid_ref"]] == f["numbers/grid"], "grid_ref attr")
    else:
        # The reference library cannot dereference object references in
        # files with offsets narrower than 8 bytes, not even in files it
        # wrote itself (observed with HDF5 2.0.0), so skip those checks.
        print("note: skipping object reference checks (size of offsets != 8)")
    info = h5py.h5o.get_info(f["numbers/grid"].id)
    check(info.rc == 2, f"grid reference count {info.rc}")


def check_many(f):
    m = f["many"]
    eq(sorted(m.keys()), [f"g{i:02}" for i in range(20)] + ["huge"], "many members")
    eq(sorted(m.attrs.keys()), [f"a{i:02}" for i in range(20)], "many attrs")
    for i in range(20):
        eq(m.attrs[f"a{i:02}"], i, f"many@a{i:02}")
    h = m["huge"]
    check(len(h) == 1500, f"huge group has {len(h)} links")
    eq(h["item0777"][()], 777, "huge/item0777")
    n = 2000 if is_compat(f) else 20000
    eq(h.attrs["blob"], np.arange(n, dtype="f8"), "huge@blob")
    eq(s(h.attrs["note"]), "a dense group", "huge@note")
    c = f["a/b/c"]
    eq(c.attrs["depth"], 3, "a/b/c@depth")
    eq(s(c.attrs["naïve \U0001f600"]), "utf-8 attribute name", "utf-8 attr name")


def visit_all(f):
    seen = []

    def visitor(name, obj):
        seen.append(name)
        for k in obj.attrs.keys():
            obj.attrs[k]
        if isinstance(obj, h5py.Dataset):
            if obj.id.get_type().get_class() == h5py.h5t.OPAQUE:
                read_opaque(obj)
            else:
                obj[()]

    f.visititems(visitor)
    for k in f.attrs.keys():
        f.attrs[k]
    return len(seen)


CHUNK_DATA = np.array([(i % 37) * 3 for i in range(1000)], dtype="i4")


def lzf_mixed_values():
    x = 0x12345678
    out = []
    for _ in range(64):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out.append(x)
    return np.array(out + [7] * 64, dtype="u4")


def hyper_values():
    h = np.zeros((20, 30), dtype="u2")
    vals = iter(range(1000, 1060))
    for r in (1, 2, 5, 6, 9, 10):
        for c in range(3, 13):
            h[r, c] = next(vals)
    h[19, :] = np.arange(30)
    return h


def log_values():
    v = np.arange(700, dtype="i4")
    return np.concatenate([v, np.full(10, -1, dtype="i4")])


def chunk_masks(d):
    """Filter masks of every stored chunk, in storage order."""
    n = d.id.get_num_chunks()
    return [d.id.get_chunk_info(i).filter_mask for i in range(n)]


def check_chunked(f):
    g = f["chunked"]
    iota = lambda n: np.arange(n, dtype="i4")  # noqa: E731

    def ds(name, shape, chunks, maxshape=None, values=None, compression=None):
        d = g[name]
        check(d.shape == shape, f"chunked/{name} shape {d.shape} != {shape}")
        check(d.chunks == chunks, f"chunked/{name} chunks {d.chunks} != {chunks}")
        want_max = maxshape if maxshape is not None else shape
        if f.id.get_create_plist().get_sizes()[1] == 4:
            # With 4-byte lengths the reference library reads the all-ones
            # "unlimited" pattern back as 2**32 - 1 (also in files it
            # writes itself).
            want_max = tuple(2**32 - 1 if m is None else m for m in want_max)
        check(d.maxshape == want_max, f"chunked/{name} maxshape {d.maxshape} != {want_max}")
        if compression is not False:
            check(d.compression == compression,
                  f"chunked/{name} compression {d.compression} != {compression}")
        if values is not None:
            eq(d[()], values, f"chunked/{name} values")
        return d

    ds("single", (10,), (10,), values=iota(10))
    ds("farray", (6, 10), (4, 4), values=iota(60).reshape(6, 10))
    ds("farray_paged", (5001,), (2,), values=iota(5001))
    d = ds("implicit", (5, 7), (2, 3), values=iota(35).reshape(5, 7))
    check(d.id.get_create_plist().get_alloc_time() == h5py.h5d.ALLOC_TIME_EARLY,
          "implicit alloc time early")
    ds("earray", (3, 700, 2), (2, 1, 2), (3, None, 4), iota(4200).reshape(3, 700, 2))
    ds("btree2", (60, 70), (1, 1), (None, None), iota(4200).reshape(60, 70))
    ds("single_gz", (10,), (10,), values=iota(10), compression="gzip")
    ds("farray_gz", (5001,), (2,), values=iota(5001), compression="gzip")
    ds("earray_gz", (20000,), (4,), (None,), iota(20000), compression="gzip")
    d = ds("btree2_gz", (40, 50), (1, 1), (None, None), iota(2000).reshape(40, 50),
           compression="gzip")
    check(d.fletcher32, "btree2_gz fletcher32")
    d = ds("gzip", (1000,), (128,), values=CHUNK_DATA, compression="gzip")
    check(d.compression_opts == 6, f"gzip level {d.compression_opts}")
    check(all(c.size < 512 for c in
              (d.id.get_chunk_info(i) for i in range(d.id.get_num_chunks()))),
          "gzip chunks are compressed")
    d = ds("shuf_gzip_f32", (1000,), (100,), values=CHUNK_DATA, compression="gzip")
    check(d.shuffle and d.fletcher32 and d.compression_opts == 9,
          f"shuf_gzip_f32 filters {d.shuffle} {d.fletcher32} {d.compression_opts}")
    ds("lzf", (1000,), (256,), values=CHUNK_DATA, compression="lzf")
    d = ds("lzf_mixed", (128,), (64,), values=lzf_mixed_values(), compression="lzf")
    eq(chunk_masks(d), [1, 0], "lzf_mixed filter masks (optional filter skipped)")
    # h5py names no compression for N-bit (it reports "unknown").
    d = ds("nbit", (1000,), (100,), values=np.arange(-500, 500, dtype="i2"),
           compression=False)
    check(d.id.get_create_plist().get_filter(0)[0] == h5py.h5z.FILTER_NBIT, "nbit filter")
    d = ds("so_int", (1000,), (64,), values=CHUNK_DATA)
    check(d.scaleoffset == 0, f"so_int scaleoffset {d.scaleoffset}")
    d = ds("so_float", (300,), (50,), values=np.arange(300, dtype="f4") * 0.25)
    check(d.scaleoffset == 2, f"so_float scaleoffset {d.scaleoffset}")
    d = ds("partial", (10,), (4,), values=iota(10), compression="gzip")
    eq(chunk_masks(d)[-1], 1, "partial edge chunk stored unfiltered")
    ds("edge2d", (25, 40), (7, 9), values=iota(1000).reshape(25, 40), compression="gzip")
    ds("auto", (2,), (1024,), (None,), [1.0, 2.0], compression="gzip")
    d = ds("log", (710,), (16,), (None,), log_values(), compression="gzip")
    eq(d.fillvalue, -1, "log fill value")
    t = np.array([[r, r + 0.5, -r] for r in range(10)], dtype="f8")
    ds("table", (10, 3), (4, 3), (None, 3), t)
    ds("empty_ext", (0, 4), (2, 2), (None, None), np.zeros((0, 4), dtype="i4"))
    ds("fill_early", (9,), (4,), values=np.full(9, -5, dtype="i2"))
    ds("hyper", (20, 30), (6, 7), values=hyper_values(), compression="gzip")
    d = ds("names", (4,), (2,), (None,))
    eq([s(x) for x in d[()]], ["a", "bb", "ccc", "dddd"], "chunked/names values")


def modify_chunked(f):
    """Resize, append and overwrite our chunked datasets with the
    reference library."""
    g = f["chunked"]
    d = g["log"]
    d.resize((720,))
    d[710:720] = np.arange(10) + 5000
    t = g["table"]
    t.resize((12, 3))
    t[10:12] = [[10, 10.5, -10], [11, 11.5, -11]]
    e = g["earray"]
    e.resize((3, 710, 2))
    e[:, 700:710, :] = -np.ones((3, 10, 2))
    b = g["btree2"]
    b.resize((62, 72))
    b[60:62, :] = 7
    b[:60, 70:72] = 8
    x = g["empty_ext"]
    x.resize((5, 6))
    x[...] = np.arange(30).reshape(5, 6)
    g["farray"][0, 0] = 99
    g["single_gz"][0] = -9
    g["farray_gz"][4000:4010] = -9
    eg = g["earray_gz"]
    eg.resize((40000,))
    eg[20000:] = -np.arange(20000)
    bg = g["btree2_gz"]
    bg.resize((41, 50))
    bg[40, :] = -2
    g["gzip"][10:20] = -3
    g["implicit"][4, 6] = -1
    g["single"][3] = 33
    n = g["names"]
    n.resize((5,))
    n[4] = "eeeee"
    g.create_dataset("by_h5py", data=np.arange(100), chunks=(7,), maxshape=(None,),
                     compression="gzip")


def check_modified_chunked(f):
    g = f["chunked"]
    want = np.concatenate([log_values(), np.arange(10) + 5000])
    eq(g["log"][()], want, "modified log")
    t = np.array([[r, r + 0.5, -r] for r in range(12)], dtype="f8")
    eq(g["table"][()], t, "modified table")
    e = np.arange(4200, dtype="i4").reshape(3, 700, 2)
    e = np.concatenate([e, -np.ones((3, 10, 2), dtype="i4")], axis=1)
    eq(g["earray"][()], e, "modified earray")
    b = np.zeros((62, 72), dtype="i4")
    b[:60, :70] = np.arange(4200).reshape(60, 70)
    b[60:62, :] = 7
    b[:60, 70:72] = 8
    eq(g["btree2"][()], b, "modified btree2")
    eq(g["empty_ext"][()], np.arange(30).reshape(5, 6), "modified empty_ext")
    fa = np.arange(60).reshape(6, 10)
    fa[0, 0] = 99
    eq(g["farray"][()], fa, "modified farray")
    gz = CHUNK_DATA.copy()
    gz[10:20] = -3
    eq(g["gzip"][()], gz, "modified gzip")
    im = np.arange(35).reshape(5, 7)
    im[4, 6] = -1
    eq(g["implicit"][()], im, "modified implicit")
    si = np.arange(10)
    si[3] = 33
    eq(g["single"][()], si, "modified single")
    eq([s(x) for x in g["names"][()]], ["a", "bb", "ccc", "dddd", "eeeee"], "modified names")
    eq(g["by_h5py"][()], np.arange(100), "dataset created by h5py")
    sg = np.arange(10)
    sg[0] = -9
    eq(g["single_gz"][()], sg, "modified single_gz")
    fg = np.arange(5001)
    fg[4000:4010] = -9
    eq(g["farray_gz"][()], fg, "modified farray_gz")
    eq(g["earray_gz"][()], np.concatenate([np.arange(20000), -np.arange(20000)]),
       "modified earray_gz")
    bg = np.concatenate([np.arange(2000).reshape(40, 50), np.full((1, 50), -2)])
    eq(g["btree2_gz"][()], bg, "modified btree2_gz")


def check_modify(path):
    """Modify a copy with the reference library (exercises its view of our
    metadata: phase-change values, reference counts, free space) and check
    the result."""
    import os
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        copy = os.path.join(tmp, "copy.h5")
        shutil.copyfile(path, copy)
        with h5py.File(copy, "r+") as f:
            m = f["many"]
            for i in range(20, 30):
                m.attrs[f"a{i:02}"] = i
                m.create_group(f"g{i:02}")
            f["numbers/grid"].attrs["added"] = "yes"
            f.create_dataset("new/data", data=np.arange(5))
            del f["links/dangling"]
            modify_chunked(f)
            f["a/b/c"].attrs["depth"] = 4
            h = f["many/huge"]
            h.create_group("added")
            del h["item0001"]
            h.attrs["blob2"] = np.ones(1000)
            del h.attrs["note"]
            for i in range(27):  # dense -> compact conversion below 6
                del m.attrs[f"a{i:02}"]
            for i in range(1500):  # empties our dense heap / B-tree
                if i != 1:
                    del h[f"item{i:04}"]
        with h5py.File(copy, "r") as f:
            m = f["many"]
            eq(sorted(m.attrs.keys()), [f"a{i:02}" for i in range(27, 30)],
               "modified attrs")
            h = f["many/huge"]
            check(list(h.keys()) == ["added"], f"modified dense group {list(h)[:5]}")
            n = 2000 if is_compat(f) else 20000
            eq(h.attrs["blob"], np.arange(n, dtype="f8"), "huge@blob kept")
            eq(h.attrs["blob2"], np.ones(1000), "huge@blob2 added")
            check("note" not in h.attrs, "huge@note deleted")
            eq(sorted(m.keys()), [f"g{i:02}" for i in range(30)] + ["huge"],
               "modified groups")
            eq(s(f["numbers/grid"].attrs["added"]), "yes", "added attr")
            eq(f["new/data"][()], np.arange(5), "new dataset")
            check("dangling" not in f["links"], "deleted link")
            eq(f["a/b/c"].attrs["depth"], 4, "rewritten attr")
            check_modified_chunked(f)
            visit_all(f)


def main(path):
    with h5py.File(path, "r") as f:
        sb = f.id.get_create_plist().get_version()[0]
        print(f"{path}: superblock v{sb}, libhdf5 {h5py.version.hdf5_version}")
        for fn in (check_root, check_numbers, check_strings, check_types,
                   check_links, check_many, check_chunked):
            try:
                fn(f)
            except Exception as e:  # noqa: BLE001 - report and continue
                check(False, f"{fn.__name__} raised {type(e).__name__}: {e}")
        try:
            n = visit_all(f)
            print(f"visited {n} objects")
        except Exception as e:  # noqa: BLE001
            check(False, f"visit raised {type(e).__name__}: {e}")
    try:
        check_modify(path)
        print("modification by the reference library: ok")
    except Exception as e:  # noqa: BLE001
        check(False, f"modify raised {type(e).__name__}: {e}")
    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(max(main(p) for p in sys.argv[1:]))
