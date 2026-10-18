#!/usr/bin/env python3
"""Generate the hdf5rs interoperability fixture corpus.

The reference implementation (libhdf5 via h5py) is used strictly as a black
box: we create files through the public h5py API, then reopen them with h5py
and record what the reference reader reports into ``manifest.json``.

Usage (see tools/README.md):

    python tools/gen_fixtures.py [--out tests/fixtures]

Output is deterministic for a given h5py/HDF5 version (no timestamps are
stored; random data uses a fixed seed).
"""

import argparse
import hashlib
import json
import math
import os
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401  (registers extra filters)
except ImportError:  # optional
    hdf5plugin = None

OUT = None
MANIFEST = {}
# Per-file, per-path extra expectations merged into the manifest
# (e.g. the chunk index type the spec rules imply).
EXPECT = {}
FILE_NOTES = {}
SUMMARY_THRESHOLD = 1024  # elements; above this only a summary is recorded


def path(name):
    return os.path.join(OUT, name)


def new_file(name, libver, description, **kw):
    p = path(name)
    if os.path.exists(p):
        os.remove(p)
    FILE_NOTES[name] = {"description": description, "libver": libver_str(libver)}
    return h5py.File(p, "w", libver=libver, **kw)


def libver_str(lv):
    if isinstance(lv, tuple):
        return list(lv)
    return lv


def expect(fname, opath, **kw):
    EXPECT.setdefault(fname, {}).setdefault(opath, {}).update(kw)


def rng():
    return np.random.default_rng(12345)


def low_level_dataset(parent, name, tid, shape, maxshape=None, dcpl=None):
    """Create a dataset with an explicit HDF5 file datatype."""
    if dcpl is None:
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_obj_track_times(False)  # no timestamps: keep output deterministic
    if shape is None:
        space = h5py.h5s.create(h5py.h5s.NULL)
    elif shape == ():
        space = h5py.h5s.create(h5py.h5s.SCALAR)
    else:
        space = h5py.h5s.create_simple(shape, maxshape)
    if "/" in name:
        parent = parent.require_group(name.rsplit("/", 1)[0])
        name = name.rsplit("/", 1)[1]
    dsid = h5py.h5d.create(parent.id, name.encode(), tid, space, dcpl=dcpl)
    return h5py.Dataset(dsid)


def compact_dataset(parent, name, data):
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_layout(h5py.h5d.COMPACT)
    tid = h5py.h5t.py_create(data.dtype, logical=True)
    d = low_level_dataset(parent, name, tid, data.shape, dcpl=dcpl)
    d[...] = data
    return d


def write_raw(dset, arr, mtype):
    """Write a numpy buffer using a memory type identical to the file type."""
    arr = np.ascontiguousarray(arr)
    dset.id.write(h5py.h5s.ALL, h5py.h5s.ALL, arr, mtype=mtype)


def str_type(size, cset, pad):
    t = h5py.h5t.C_S1.copy()
    t.set_size(size)
    t.set_cset(cset)
    t.set_strpad(pad)
    return t


# ---------------------------------------------------------------------------
# Fixture generators
# ---------------------------------------------------------------------------


def gen_superblocks():
    for name, lv, desc in [
        ("superblock_v0.h5", "earliest", "libver=earliest: v0 superblock, v1 object headers, symbol-table groups"),
        ("superblock_v2.h5", ("v108", "v108"), "libver=v108: v2 superblock, v2 object headers"),
        ("superblock_v3.h5", "latest", "libver=latest: v3 superblock, v2 object headers, link messages"),
    ]:
        with new_file(name, lv, desc) as f:
            f.attrs["title"] = "superblock test"
            f.create_dataset("data", data=np.arange(10, dtype="<i4"))
            g = f.create_group("group")
            g.create_dataset("values", data=np.linspace(0.0, 1.0, 5))
            g.attrs["n"] = np.int64(5)

    name = "userblock.h5"
    with new_file(name, "earliest", "512-byte userblock before the superblock (v0 superblock)", userblock_size=512) as f:
        f.create_dataset("data", data=np.arange(4, dtype="<u2"))
    ub = b"hdf5rs userblock test\n".ljust(512, b"\0")
    with open(path(name), "r+b") as fh:
        fh.write(ub)
    FILE_NOTES[name]["userblock_size"] = 512
    FILE_NOTES[name]["userblock_prefix"] = "hdf5rs userblock test\n"


def fill_groups(f, latest):
    a = f.create_group("a")
    b = a.create_group("b")
    c = b.create_group("c")
    c.create_dataset("leaf", data=np.arange(3, dtype="<i8"))
    f.create_group("empty")
    many = f.create_group("many")
    for i in range(100):
        many.create_dataset("d%03d" % i, data=np.int32(i))
    d = f.create_dataset("target", data=np.arange(6, dtype="<i4").reshape(2, 3))
    f["hard_alias"] = d  # second hard link to the same dataset
    a["hard_alias2"] = d
    f["soft_to_target"] = h5py.SoftLink("/target")
    f["soft_relative"] = h5py.SoftLink("a/b/c/leaf")
    f["soft_dangling"] = h5py.SoftLink("/does/not/exist")
    f["external"] = h5py.ExternalLink("superblock_v3.h5", "/data")
    if latest:
        t = f.create_group("tracked", track_order=True)
        for nm in ["zeta", "alpha", "mid", "beta"]:
            t.create_group(nm)
        t2 = f.create_group("tracked_many", track_order=True)
        for i in range(30):
            t2.create_dataset("d%02d" % (29 - i), data=np.int32(i))


def gen_groups():
    with new_file("groups_earliest.h5", "earliest",
                  "Old-style groups: nested, empty, 100-entry symbol table (multi-node B-tree), soft/hard/external links") as f:
        fill_groups(f, False)
    with new_file("groups_latest.h5", "latest",
                  "New-style groups: nested, empty, 100 links (dense storage), creation-order tracked groups, soft/hard/external links") as f:
        fill_groups(f, True)
    expect("groups_latest.h5", "/many", link_storage="dense")
    expect("groups_latest.h5", "/tracked", link_storage="compact", creation_order_tracked=True)
    expect("groups_latest.h5", "/tracked_many", link_storage="dense", creation_order_tracked=True)
    expect("groups_latest.h5", "/a", link_storage="compact")


def fill_dtypes(f, latest):
    n = 5
    for bits in (8, 16, 32, 64):
        for kind in ("i", "u"):
            for order in ("<", ">"):
                dt = np.dtype("%s%s%d" % (order, kind, bits // 8))
                info = np.iinfo(dt)
                vals = np.array([info.min, info.max, 0, 1, info.max // 3], dtype=dt)
                f.create_dataset("int/%s%d%s" % (kind, bits, "le" if order == "<" else "be"), data=vals)
    for bits in (16, 32, 64):
        for order in ("<", ">"):
            dt = np.dtype("%sf%d" % (order, bits // 8))
            vals = np.array([0.0, -1.5, 3.25, np.inf, -np.inf, np.nan, np.finfo(dt).max, np.finfo(dt).tiny], dtype=dt)
            f.create_dataset("float/f%d%s" % (bits, "le" if order == "<" else "be"), data=vals)
    f.create_dataset("bool", data=np.array([True, False, True, True, False]))

    # complex numbers (mapped by h5py to whatever HDF5 type it uses)
    f.create_dataset("complex/c64", data=np.array([1 + 2j, -3.5 - 0.25j], dtype="<c8"))
    f.create_dataset("complex/c128", data=np.array([1 + 2j, -3.5 - 0.25j], dtype="<c16"))

    # fixed-length strings
    words = [b"alpha", b"be", b"", b"gammadel"]
    for cset_name, cset in (("ascii", h5py.h5t.CSET_ASCII), ("utf8", h5py.h5t.CSET_UTF8)):
        for pad_name, pad in (("nullterm", h5py.h5t.STR_NULLTERM), ("nullpad", h5py.h5t.STR_NULLPAD),
                              ("spacepad", h5py.h5t.STR_SPACEPAD)):
            t = str_type(8, cset, pad)
            d = low_level_dataset(f, "string/fixed_%s_%s" % (cset_name, pad_name), t, (len(words),))
            buf = np.array(words, dtype="S8")
            if pad == h5py.h5t.STR_SPACEPAD:
                buf = np.array([w.ljust(8, b" ") for w in words], dtype="S8")
            write_raw(d, buf, t)
    utf = ["héllo".encode(), "日本".encode(), b"x"]
    t = str_type(10, h5py.h5t.CSET_UTF8, h5py.h5t.STR_NULLPAD)
    d = low_level_dataset(f, "string/fixed_utf8_multibyte", t, (3,))
    write_raw(d, np.array(utf, dtype="S10"), t)

    f.create_dataset("string/vlen_utf8", data=["", "short", "a longer variable length string", "日本語テキスト"],
                     dtype=h5py.string_dtype("utf-8"))
    f.create_dataset("string/vlen_ascii", data=[b"one", b"two", b"three"], dtype=h5py.string_dtype("ascii"))
    f.create_dataset("string/vlen_scalar", data="scalar vlen string", dtype=h5py.string_dtype("utf-8"))
    f.create_dataset("string/fixed_scalar", data=np.bytes_(b"fixed!"))

    vl = f.create_dataset("vlen/int32", (4,), dtype=h5py.vlen_dtype(np.dtype("<i4")))
    vl[0] = np.array([], dtype="<i4")
    vl[1] = np.array([1], dtype="<i4")
    vl[2] = np.array([1, 2, 3], dtype="<i4")
    vl[3] = np.arange(10, dtype="<i4")
    vf = f.create_dataset("vlen/float64", (2,), dtype=h5py.vlen_dtype(np.dtype("<f8")))
    vf[0] = np.array([0.5, 1.5])
    vf[1] = np.array([-2.0])

    inner = np.dtype([("x", "<f8"), ("y", "u1")])
    comp = np.dtype([("a", "<i4"), ("b", inner), ("c", "<i2", (3,)), ("s", "S5"), ("e", ">u2")])
    cdata = np.zeros(3, dtype=comp)
    for i in range(3):
        cdata[i] = (i - 1, (i * 0.5, 200 + i), (i, i + 1, i + 2), b"s%d" % i, 1000 + i)
    f.create_dataset("compound/nested", data=cdata)
    simple = np.dtype([("id", "<u4"), ("val", "<f4")])
    f.create_dataset("compound/simple", data=np.array([(1, 1.5), (2, -2.5)], dtype=simple))
    padded = np.dtype({"names": ["a", "b"], "formats": ["u1", "<i4"], "offsets": [0, 8], "itemsize": 16})
    pd = np.zeros(2, dtype=padded)
    pd["a"] = [7, 8]
    pd["b"] = [-7, -8]
    f.create_dataset("compound/padded", data=pd)
    vcomp = np.dtype([("name", h5py.string_dtype("utf-8")), ("vals", h5py.vlen_dtype(np.dtype("<i2")))])
    vc = f.create_dataset("compound/with_vlen", (2,), dtype=vcomp)
    vc[0] = ("first", np.array([1, 2], dtype="<i2"))
    vc[1] = ("second", np.array([], dtype="<i2"))

    e8 = h5py.enum_dtype({"RED": 0, "GREEN": 1, "BLUE": 42}, basetype="i1")
    f.create_dataset("enum/int8", data=np.array([0, 1, 42, 1], dtype="i1"), dtype=e8)
    e16 = h5py.enum_dtype({"LOW": -100, "HIGH": 1000}, basetype=">i2")
    f.create_dataset("enum/int16be", data=np.array([-100, 1000], dtype=">i2"), dtype=e16)

    at = h5py.h5t.array_create(h5py.h5t.STD_I32LE, (2, 3))
    d = low_level_dataset(f, "array/i32_2x3", at, (2,))
    write_raw(d, np.arange(12, dtype="<i4").reshape(2, 2, 3), at)
    at2 = h5py.h5t.array_create(h5py.h5t.IEEE_F64BE, (4,))
    d = low_level_dataset(f, "array/f64be_4", at2, (3,))
    write_raw(d, (np.arange(12).reshape(3, 4) / 4).astype(">f8"), at2)

    f.create_dataset("opaque/v8", data=np.frombuffer(bytes(range(24)), dtype="V8"))
    ot = h5py.h5t.create(h5py.h5t.OPAQUE, 4)
    ot.set_tag(b"hdf5rs-tag")
    d = low_level_dataset(f, "opaque/tagged", ot, (2,))
    write_raw(d, np.frombuffer(b"\xde\xad\xbe\xef\x01\x02\x03\x04", dtype="V4"), ot)

    for name, t in (("bitfield/b8", h5py.h5t.STD_B8LE), ("bitfield/b16be", h5py.h5t.STD_B16BE),
                    ("bitfield/b32", h5py.h5t.STD_B32LE)):
        t = t.copy()
        sz = t.get_size()
        raw = bytes((i * 37 + 5) & 0xFF for i in range(4 * sz))
        d = low_level_dataset(f, name, t, (4,))
        write_raw(d, np.frombuffer(raw, dtype="V%d" % sz), t)

    # named (committed) datatypes
    f.create_group("named")  # explicit: implicit intermediate groups get timestamps
    f["named/point"] = np.dtype([("x", "<f4"), ("y", "<f4")])
    f["named/uint16be"] = np.dtype(">u2")
    nt = f["named/point"]
    d = f.create_dataset("named_use/points", (3,), dtype=nt)
    d[...] = np.array([(0, 0), (1, 2), (3.5, -4)], dtype=nt.dtype)
    d.attrs.create("origin", np.array((0.5, 0.25), dtype=nt.dtype), dtype=nt)
    d2 = f.create_dataset("named_use/u16", (2,), dtype=f["named/uint16be"])
    d2[...] = [1, 65535]

    # references
    tgt = f.create_dataset("ref/target", data=np.arange(20, dtype="<i4").reshape(4, 5))
    g = f.create_group("ref/group")
    refs = f.create_dataset("ref/object", (4,), dtype=h5py.ref_dtype)
    refs[0] = tgt.ref
    refs[1] = g.ref
    refs[2] = f.ref
    # refs[3] stays a null reference
    rr = f.create_dataset("ref/region", (3,), dtype=h5py.regionref_dtype)
    rr[0] = tgt.regionref[1:3, 0:2]
    rr[1] = tgt.regionref[0, :]
    rr[2] = tgt.regionref[...]
    f.attrs["ref_to_target"] = tgt.ref


def gen_dtypes():
    with new_file("dtypes_earliest.h5", "earliest", "Every datatype class, earliest format") as f:
        fill_dtypes(f, False)
    with new_file("dtypes_latest.h5", "latest", "Every datatype class, latest format") as f:
        fill_dtypes(f, True)


def gen_dataspaces():
    with new_file("dataspaces.h5", "latest", "Scalar, null, 1-D..4-D, zero-size and unlimited dataspaces") as f:
        f.create_dataset("scalar", data=np.float64(42.5))
        f.create_dataset("scalar_int", data=np.int16(-7))
        f.create_dataset("null", data=h5py.Empty("<f4"))
        f.attrs["null_attr"] = h5py.Empty("<i4")
        f.create_dataset("d1", data=np.arange(7, dtype="<i2"))
        f.create_dataset("d2", data=np.arange(12, dtype="<i2").reshape(3, 4))
        f.create_dataset("d3", data=np.arange(24, dtype="<i2").reshape(2, 3, 4))
        f.create_dataset("d4", data=np.arange(120, dtype="<i2").reshape(2, 3, 4, 5))
        f.create_dataset("zero_1d", shape=(0,), dtype="<f8")
        f.create_dataset("zero_2d", shape=(3, 0), dtype="<f8")
        f.create_dataset("zero_unlimited", shape=(0,), maxshape=(None,), dtype="<i4", chunks=(4,))
        f.create_dataset("unlimited_1d", data=np.arange(5, dtype="<i4"), maxshape=(None,), chunks=(2,))
        f.create_dataset("unlimited_2d", data=np.arange(6, dtype="<i4").reshape(2, 3), maxshape=(None, 10), chunks=(1, 2))
        f.create_dataset("bounded_max", data=np.arange(4, dtype="<i4"), maxshape=(8,), chunks=(4,))
    with new_file("dataspaces_earliest.h5", "earliest", "Scalar, null, simple and unlimited dataspaces, earliest format") as f:
        f.create_dataset("scalar", data=np.float64(42.5))
        f.create_dataset("null", data=h5py.Empty("<f4"))
        f.create_dataset("d3", data=np.arange(24, dtype="<i2").reshape(2, 3, 4))
        f.create_dataset("zero_1d", shape=(0,), dtype="<f8")
        f.create_dataset("unlimited_2d", data=np.arange(6, dtype="<i4").reshape(2, 3), maxshape=(None, None), chunks=(1, 2))


def gen_layouts():
    fn = "layouts_latest.h5"
    with new_file(fn, "latest", "Compact, contiguous and every v4 chunk index (single, implicit, fixed array, extensible array, v2 B-tree), with partial chunks") as f:
        compact_dataset(f, "compact", np.arange(10, dtype="<i4"))
        f.create_dataset("contiguous", data=np.arange(100, dtype="<f4").reshape(10, 10))
        f.create_dataset("contiguous_unwritten", shape=(10,), dtype="<i4", fillvalue=-5)
        expect(fn, "/contiguous_unwritten", storage_allocated=False)
        f.create_dataset("chunk_single", data=np.arange(20, dtype="<i4").reshape(4, 5), chunks=(4, 5))
        expect(fn, "/chunk_single", chunk_index="single_chunk")
        f.create_dataset("chunk_single_gzip", data=np.arange(20, dtype="<i4").reshape(4, 5), chunks=(4, 5), compression="gzip")
        expect(fn, "/chunk_single_gzip", chunk_index="single_chunk")
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((3, 4))
        dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
        d = low_level_dataset(f, "chunk_implicit", h5py.h5t.STD_I32LE, (10, 10), dcpl=dcpl)
        d[...] = np.arange(100, dtype="<i4").reshape(10, 10)
        expect(fn, "/chunk_implicit", chunk_index="implicit")
        f.create_dataset("chunk_fixed_array", data=np.arange(100, dtype="<i4").reshape(10, 10), chunks=(3, 4))
        expect(fn, "/chunk_fixed_array", chunk_index="fixed_array")
        f.create_dataset("chunk_fixed_array_paged", data=(np.arange(2000) % 251).astype("u1"), chunks=(1,))
        expect(fn, "/chunk_fixed_array_paged", chunk_index="fixed_array", note="2000 chunks: more than one fixed-array data block page")
        f.create_dataset("chunk_fixed_array_gzip", data=np.arange(100, dtype="<i4").reshape(10, 10), chunks=(3, 4), compression="gzip")
        expect(fn, "/chunk_fixed_array_gzip", chunk_index="fixed_array")
        d = f.create_dataset("chunk_fixed_array_partial", shape=(10, 10), dtype="<i4", chunks=(3, 4), fillvalue=-1)
        d[0:3, 0:4] = 1
        d[9, 9] = 2
        expect(fn, "/chunk_fixed_array_partial", chunk_index="fixed_array")
        d = f.create_dataset("chunk_extensible_array", shape=(0,), maxshape=(None,), dtype="<i4", chunks=(4,))
        d.resize((50,))
        d[...] = np.arange(50, dtype="<i4")
        expect(fn, "/chunk_extensible_array", chunk_index="extensible_array")
        d = f.create_dataset("chunk_extensible_array_big", data=np.arange(800, dtype="<i2"), maxshape=(None,), chunks=(2,))
        expect(fn, "/chunk_extensible_array_big", chunk_index="extensible_array", note="400 chunks: uses EA secondary/data blocks")
        d = f.create_dataset("chunk_extensible_array_partial", shape=(40,), maxshape=(None,), dtype="<i4", chunks=(4,), fillvalue=-1)
        d[0:4] = 7
        d[30:34] = 8
        expect(fn, "/chunk_extensible_array_partial", chunk_index="extensible_array")
        d = f.create_dataset("chunk_extensible_array_gzip", data=np.arange(50, dtype="<i4"), maxshape=(None,), chunks=(8,), compression="gzip")
        expect(fn, "/chunk_extensible_array_gzip", chunk_index="extensible_array")
        d = f.create_dataset("chunk_btree_v2", data=np.arange(30, dtype="<i4").reshape(5, 6), maxshape=(None, None), chunks=(2, 2))
        expect(fn, "/chunk_btree_v2", chunk_index="btree_v2")
        d = f.create_dataset("chunk_btree_v2_big", data=np.arange(400, dtype="<i2").reshape(20, 20), maxshape=(None, None), chunks=(1, 1))
        expect(fn, "/chunk_btree_v2_big", chunk_index="btree_v2", note="400 chunks: multi-level v2 B-tree")
        d = f.create_dataset("chunk_btree_v2_partial", shape=(6, 6), maxshape=(None, None), dtype="<i4", chunks=(2, 2), fillvalue=-1)
        d[0:2, 0:2] = 3
        d[5, 5] = 4
        expect(fn, "/chunk_btree_v2_partial", chunk_index="btree_v2")
        d = f.create_dataset("chunk_btree_v2_gzip", data=np.arange(30, dtype="<i4").reshape(5, 6), maxshape=(None, None), chunks=(2, 2), compression="gzip")
        expect(fn, "/chunk_btree_v2_gzip", chunk_index="btree_v2")
        f.create_dataset("chunk_edge", data=np.arange(35, dtype="<f8").reshape(5, 7), chunks=(2, 3))
        expect(fn, "/chunk_edge", chunk_index="fixed_array", note="shape not a multiple of chunk shape")
        f.create_dataset("chunk_never_written", shape=(8, 8), dtype="<f4", chunks=(4, 4), fillvalue=1.5)
        expect(fn, "/chunk_never_written", chunk_index="fixed_array")
        f.create_dataset("chunk_3d", data=np.arange(60, dtype="<u2").reshape(3, 4, 5), chunks=(2, 2, 2))
        expect(fn, "/chunk_3d", chunk_index="fixed_array")

    fn = "layouts_earliest.h5"
    with new_file(fn, "earliest", "Compact, contiguous and v1 B-tree chunked datasets (layout message v3)") as f:
        compact_dataset(f, "compact", np.arange(10, dtype="<i4"))
        f.create_dataset("contiguous", data=np.arange(100, dtype="<f4").reshape(10, 10))
        f.create_dataset("contiguous_unwritten", shape=(10,), dtype="<i4", fillvalue=-5)
        f.create_dataset("chunked", data=np.arange(100, dtype="<i4").reshape(10, 10), chunks=(3, 4))
        f.create_dataset("chunked_gzip", data=np.arange(100, dtype="<i4").reshape(10, 10), chunks=(3, 4), compression="gzip")
        f.create_dataset("chunked_many", data=np.arange(800, dtype="<i2"), chunks=(2,))
        d = f.create_dataset("chunked_partial", shape=(10, 10), dtype="<i4", chunks=(3, 4), fillvalue=-1)
        d[0:3, 0:4] = 1
        d[9, 9] = 2
        f.create_dataset("chunked_unlimited", data=np.arange(30, dtype="<i4").reshape(5, 6), maxshape=(None, None), chunks=(2, 2))
        for p in ("chunked", "chunked_gzip", "chunked_many", "chunked_partial", "chunked_unlimited"):
            expect(fn, "/" + p, chunk_index="btree_v1")
        expect(fn, "/chunked_many", note="400 chunks: multi-level v1 B-tree")


def gen_filters():
    fn = "filters.h5"
    base = np.arange(1000, dtype="<i4").reshape(20, 50)
    fl = np.linspace(-10, 10, 1000).reshape(20, 50)
    with new_file(fn, "latest", "Filter pipelines: deflate levels, shuffle, fletcher32, lzf, scaleoffset, nbit") as f:
        for lvl in (1, 4, 9):
            f.create_dataset("gzip_%d" % lvl, data=base, chunks=(5, 10), compression="gzip", compression_opts=lvl)
        f.create_dataset("shuffle_gzip", data=base, chunks=(5, 10), shuffle=True, compression="gzip")
        f.create_dataset("shuffle_only", data=base, chunks=(5, 10), shuffle=True)
        f.create_dataset("fletcher32", data=base, chunks=(5, 10), fletcher32=True)
        f.create_dataset("shuffle_gzip_fletcher32", data=fl, chunks=(5, 10), shuffle=True, compression="gzip", fletcher32=True)
        f.create_dataset("lzf", data=base, chunks=(5, 10), compression="lzf")
        f.create_dataset("lzf_float", data=fl, chunks=(5, 10), compression="lzf")
        f.create_dataset("scaleoffset_int", data=base, chunks=(5, 10), scaleoffset=0)
        f.create_dataset("scaleoffset_float", data=fl, chunks=(5, 10), scaleoffset=3)
        f.create_dataset("gzip_be_int16", data=base.astype(">i2"), chunks=(5, 10), compression="gzip")
        f.create_dataset("gzip_random", data=rng().integers(0, 2**31, size=1000, dtype="<u4"), chunks=(100,), compression="gzip")
        f.create_dataset("gzip_vlen_str", data=["v%d" % i for i in range(30)], dtype=h5py.string_dtype(), chunks=(8,), compression="gzip")
        # nbit: 12-bit precision unsigned integer stored as u16
        t = h5py.h5t.STD_U16LE.copy()
        t.set_precision(12)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((50,))
        dcpl.set_filter(5, 0)  # N-bit (H5Z_FILTER_NBIT)
        d = low_level_dataset(f, "nbit_u12", t, (200,), dcpl=dcpl)
        d[...] = (np.arange(200) * 20 % 4096).astype("<u2")
        t = h5py.h5t.STD_I32LE.copy()
        t.set_precision(20)
        t.set_offset(4)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((50,))
        dcpl.set_filter(5, 0)  # N-bit (H5Z_FILTER_NBIT)
        d = low_level_dataset(f, "nbit_i20_offset4", t, (200,), dcpl=dcpl)
        d[...] = (np.arange(200, dtype="<i4") - 100) * 1000

    fn = "filters_earliest.h5"
    with new_file(fn, "earliest", "Filter pipelines with v1 B-tree chunk index and filter pipeline message v1") as f:
        f.create_dataset("gzip", data=base, chunks=(5, 10), compression="gzip")
        f.create_dataset("shuffle_gzip_fletcher32", data=fl, chunks=(5, 10), shuffle=True, compression="gzip", fletcher32=True)
        f.create_dataset("lzf", data=base, chunks=(5, 10), compression="lzf")

    if hdf5plugin is not None:
        fn = "filters_plugins.h5"
        with new_file(fn, "latest", "Third-party filters via hdf5plugin (blosc, zstd, lz4, bzip2, bitshuffle) - may be unsupported") as f:
            f.create_dataset("blosc_lz4", data=base, chunks=(5, 10), **hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
            f.create_dataset("blosc_zstd", data=base, chunks=(5, 10), **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
            f.create_dataset("zstd", data=base, chunks=(5, 10), **hdf5plugin.Zstd(clevel=3))
            f.create_dataset("lz4", data=base, chunks=(5, 10), **hdf5plugin.LZ4())
            f.create_dataset("bzip2", data=base, chunks=(5, 10), **hdf5plugin.BZip2(blocksize=9))
            f.create_dataset("bitshuffle_lz4", data=base, chunks=(5, 10), **hdf5plugin.Bitshuffle(cname="lz4"))


def fill_attrs(f, latest):
    f.attrs["root_int"] = np.int32(1)
    f.attrs["root_str"] = "root attribute"
    g = f.create_group("group")
    d = f.create_dataset("dataset", data=np.arange(4, dtype="<i4"))
    for obj in (g, d):
        a = obj.attrs
        a["i8"] = np.int8(-8)
        a["u64be"] = np.array(2**63 + 5, dtype=">u8")
        a["f32"] = np.float32(1.25)
        a["f64_array"] = np.linspace(0, 1, 6).reshape(2, 3)
        a["f16"] = np.float16(0.5)
        a["bool"] = np.bool_(True)
        a["fixed_str"] = np.bytes_(b"fixed")
        a["fixed_str_array"] = np.array([b"ab", b"cde"], dtype="S3")
        a["vlen_str"] = "variable ✓"
        a.create("vlen_str_array", ["x", "yy", "zzz"], dtype=h5py.string_dtype())
        a.create("vlen_int", np.array([np.array([1, 2], dtype="<i4"), np.array([3], dtype="<i4")], dtype=object),
                 dtype=h5py.vlen_dtype(np.dtype("<i4")))
        a["compound"] = np.array((1, 2.5), dtype=[("a", "<i2"), ("b", "<f8")])
        a.create("enum", np.int8(1), dtype=h5py.enum_dtype({"OFF": 0, "ON": 1}, basetype="i1"))
        a["opaque"] = np.void(b"\x01\x02\x03")
        a.create("array_type", np.arange(6, dtype="<i4").reshape(2, 3), dtype=np.dtype(("<i4", (3,))))
        a["empty"] = h5py.Empty("<f8")
        a["ref"] = d.ref
    if latest:
        dense = f.create_group("dense")
        for i in range(20):
            dense.attrs["attr%02d" % i] = np.int32(i)
        big = f.create_group("large")
        big.attrs["big"] = np.arange(20000, dtype="<i4")  # 80000 bytes > 64 KiB
        to = f.create_group("tracked", track_order=True)
        for nm in ["zz", "aa", "mm", "bb"]:
            to.attrs[nm] = nm
        tod = f.create_group("tracked_dense", track_order=True)
        for i in range(12):
            tod.attrs["t%02d" % (11 - i)] = np.int64(i)
        tds = f.create_dataset("tracked_dataset", data=np.int32(0), track_order=True)
        tds.attrs["second"] = 2
        tds.attrs["first"] = 1
    else:
        many = f.create_group("many")
        for i in range(20):
            many.attrs["attr%02d" % i] = np.int32(i)


def gen_attributes():
    with new_file("attributes_earliest.h5", "earliest", "Attributes of every type category on root/group/dataset, earliest format") as f:
        fill_attrs(f, False)
    with new_file("attributes_latest.h5", "latest", "Attributes: every type category, dense storage (20 attrs), >64KiB attribute, creation-order tracking") as f:
        fill_attrs(f, True)
    expect("attributes_latest.h5", "/dense", attribute_storage="dense")
    expect("attributes_latest.h5", "/large", attribute_storage="dense")
    expect("attributes_latest.h5", "/tracked", attribute_storage="compact", attr_creation_order_tracked=True)
    expect("attributes_latest.h5", "/tracked_dense", attribute_storage="dense", attr_creation_order_tracked=True)


def gen_vds():
    with new_file("vds_source.h5", "latest", "Source file for the virtual dataset in vds.h5") as f:
        f.create_dataset("src", data=np.arange(10, 20, dtype="<i4"))
    with new_file("vds.h5", "latest", "Virtual datasets: same-file source and a source in vds_source.h5") as f:
        f.create_dataset("local_src", data=np.arange(0, 10, dtype="<i4"))

        # Low-level API so that object timestamps can be disabled
        # (Group.create_virtual_dataset has no track_times option).
        def vds(name, shape, fill, mappings):
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            dcpl.set_fill_value(np.array(fill, dtype="<i4"))
            for vstart, vcount, vstride, vblock, fname, dname, sshape, ssel in mappings:
                vspace = h5py.h5s.create_simple(shape)
                vspace.select_hyperslab(vstart, vcount, vstride, vblock)
                sspace = h5py.h5s.create_simple(sshape)
                if ssel is not None:
                    sspace.select_hyperslab(*ssel)
                dcpl.set_virtual(vspace, fname.encode(), dname.encode(), sspace)
            low_level_dataset(f, name, h5py.h5t.STD_I32LE, shape, dcpl=dcpl)

        vds("virtual", (25,), -1, [
            ((0,), (1,), None, (10,), ".", "local_src", (10,), None),
            ((10,), (1,), None, (10,), "vds_source.h5", "src", (10,), None),
        ])
        vds("virtual_strided", (2, 5), 0, [
            ((0, 0), (1, 1), None, (1, 5), ".", "local_src", (10,), ((0,), (5,), (2,), (1,))),
            ((1, 0), (1, 1), None, (1, 5), ".", "local_src", (10,), ((1,), (5,), (2,), (1,))),
        ])


def gen_misc():
    with new_file("resized.h5", "latest", "Chunked datasets created then resized (grown and shrunk)") as f:
        d = f.create_dataset("grown", data=np.arange(10, dtype="<i4"), maxshape=(None,), chunks=(4,), fillvalue=-1)
        d.resize((25,))
        d[20:25] = 99
        d2 = f.create_dataset("grown_2d", data=np.arange(6, dtype="<f4").reshape(2, 3), maxshape=(None, None), chunks=(2, 2))
        d2.resize((4, 5))
        d3 = f.create_dataset("shrunk", data=np.arange(20, dtype="<i4"), maxshape=(None,), chunks=(4,))
        d3.resize((7,))
    with new_file("resized_earliest.h5", "earliest", "Chunked dataset resized, earliest format") as f:
        d = f.create_dataset("grown", data=np.arange(10, dtype="<i4"), maxshape=(None,), chunks=(4,), fillvalue=-1)
        d.resize((25,))
        d[20:25] = 99

    with new_file("big_gzip.h5", "latest", "Larger arrays of known pattern (shuffle+gzip) for performance tests") as f:
        f.create_dataset("arange_i32", data=np.arange(1 << 20, dtype="<i4"), chunks=(1 << 14,), shuffle=True, compression="gzip")
        x = np.arange(512 * 512, dtype="<f8").reshape(512, 512) * 0.5
        f.create_dataset("half_f64_2d", data=x, chunks=(64, 64), shuffle=True, compression="gzip", compression_opts=9)

    # Non-default offset/length sizes and file-space strategy (custom FCPL).
    # (SOHM tables cannot be requested through h5py's low-level API.)
    def custom_file(name, low, high, desc, setup):
        fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
        fcpl.set_obj_track_times(False)
        setup(fcpl)
        fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        fapl.set_libver_bounds(low, high)
        if os.path.exists(path(name)):
            os.remove(path(name))
        fid = h5py.h5f.create(path(name).encode(), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
        FILE_NOTES[name] = {"description": desc}
        return h5py.File(fid)

    def fill_small(f):
        f.attrs["note"] = "custom file creation properties"
        f.create_dataset("contiguous", data=np.arange(12, dtype="<i4").reshape(3, 4))
        f.create_dataset("chunked", data=np.arange(40, dtype="<i2"), chunks=(8,), compression="gzip")
        g = f.create_group("group")
        for i in range(12):
            g.create_group("sub%02d" % i)
        g.attrs["x"] = np.float64(1.5)

    E, L = h5py.h5f.LIBVER_EARLIEST, h5py.h5f.LIBVER_LATEST
    with custom_file("sizes_4byte_earliest.h5", E, L,
                     "4-byte offsets and lengths (sizeof_addr=4, sizeof_size=4), earliest format",
                     lambda p: p.set_sizes(4, 4)) as f:
        fill_small(f)
    FILE_NOTES["sizes_4byte_earliest.h5"].update(libver="earliest", sizeof_offsets=4, sizeof_lengths=4)
    with custom_file("sizes_4byte_latest.h5", L, L, "4-byte offsets and lengths, latest format",
                     lambda p: p.set_sizes(4, 4)) as f:
        fill_small(f)
    FILE_NOTES["sizes_4byte_latest.h5"].update(libver="latest", sizeof_offsets=4, sizeof_lengths=4)

    def paged(p):
        p.set_file_space_strategy(h5py.h5f.FSPACE_STRATEGY_PAGE, True, 1)
        p.set_file_space_page_size(4096)

    with custom_file("fsm_paged.h5", L, L,
                     "Paged file-space strategy with persistent free-space tracking (file space info message)",
                     paged) as f:
        fill_small(f)
        del f["group/sub03"]
        f.create_dataset("temp", data=np.arange(100, dtype="<i8"))
        del f["temp"]
    FILE_NOTES["fsm_paged.h5"].update(libver="latest")



# ---------------------------------------------------------------------------
# Manifest: describe files as reported by the reference reader
# ---------------------------------------------------------------------------

H5T = h5py.h5t
CLASS_NAMES = {
    H5T.INTEGER: "integer", H5T.FLOAT: "float", H5T.TIME: "time", H5T.STRING: "string",
    H5T.BITFIELD: "bitfield", H5T.OPAQUE: "opaque", H5T.COMPOUND: "compound",
    H5T.REFERENCE: "reference", H5T.ENUM: "enum", H5T.VLEN: "vlen", H5T.ARRAY: "array",
}
if hasattr(H5T, "COMPLEX"):
    CLASS_NAMES[H5T.COMPLEX] = "complex"


def order_str(t):
    o = t.get_order()
    return {H5T.ORDER_LE: "le", H5T.ORDER_BE: "be"}.get(o, "none")


def describe_type(t):
    cls = t.get_class()
    d = {"class": CLASS_NAMES.get(cls, "unknown(%d)" % cls)}
    if cls != H5T.VLEN and not (cls == H5T.STRING and t.is_variable_str()):
        d["size"] = t.get_size()
    if cls == H5T.INTEGER:
        d["order"] = order_str(t)
        d["signed"] = t.get_sign() == H5T.SGN_2
        if t.get_precision() != 8 * t.get_size() or t.get_offset() != 0:
            d["precision"] = t.get_precision()
            d["bit_offset"] = t.get_offset()
    elif cls == H5T.FLOAT:
        d["order"] = order_str(t)
        spos, epos, esize, mpos, msize = t.get_fields()
        d["fields"] = {"sign_pos": spos, "exp_pos": epos, "exp_size": esize, "mant_pos": mpos,
                       "mant_size": msize, "exp_bias": t.get_ebias()}
    elif cls == H5T.STRING:
        if t.is_variable_str():
            d["size"] = "variable"
        d["charset"] = {H5T.CSET_ASCII: "ascii", H5T.CSET_UTF8: "utf8"}.get(t.get_cset(), "?")
        d["padding"] = {H5T.STR_NULLTERM: "nullterm", H5T.STR_NULLPAD: "nullpad",
                        H5T.STR_SPACEPAD: "spacepad"}.get(t.get_strpad(), "?")
    elif cls == H5T.BITFIELD:
        d["order"] = order_str(t)
    elif cls == H5T.OPAQUE:
        d["tag"] = t.get_tag().decode("utf-8", "replace")
    elif cls == H5T.COMPOUND:
        d["members"] = []
        for i in range(t.get_nmembers()):
            d["members"].append({"name": t.get_member_name(i).decode(), "offset": t.get_member_offset(i),
                                 "type": describe_type(t.get_member_type(i))})
    elif cls == H5T.REFERENCE:
        if t.equal(H5T.STD_REF_OBJ):
            d["kind"] = "object"
        elif t.equal(H5T.STD_REF_DSETREG):
            d["kind"] = "region"
        else:
            d["kind"] = "other"
    elif cls == H5T.ENUM:
        d["base"] = describe_type(t.get_super())
        d["members"] = {t.get_member_name(i).decode(): int(t.get_member_value(i)) for i in range(t.get_nmembers())}
    elif cls == H5T.VLEN:
        d["base"] = describe_type(t.get_super())
    elif cls == H5T.ARRAY:
        d["dims"] = list(t.get_array_dims())
        d["base"] = describe_type(t.get_super())
    elif cls == getattr(H5T, "COMPLEX", -999):
        d["order"] = order_str(t)
        d["base"] = describe_type(t.get_super())
    if t.committed():
        d["committed"] = True
        try:
            nm = h5py.h5i.get_name(t)
            if nm:
                d["committed_path"] = nm.decode()
        except Exception:
            pass
    return d


def jfloat(x):
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def jbytes(b):
    b = bytes(b)
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return {"hex": b.hex()}


class Ctx:
    def __init__(self, f):
        self.f = f


def conv_value(x, dt, ctx):
    """Convert one element (as returned by h5py) to JSON."""
    if dt.names:
        return {n: conv_any(x[n], dt.fields[n][0], ctx) for n in dt.names}
    if dt.subdtype is not None:
        base, _ = dt.subdtype
        return conv_any(np.asarray(x), base, ctx, array=True)
    vl = h5py.check_vlen_dtype(dt)
    if vl is not None:
        if h5py.check_string_dtype(dt) is not None or vl in (str, bytes):
            return jbytes(x) if isinstance(x, (bytes, np.bytes_)) else str(x)
        return conv_any(np.asarray(x), vl, ctx, array=True)
    s = h5py.check_string_dtype(dt)
    if s is not None:
        if isinstance(x, str):
            return x
        return jbytes(x)
    r = h5py.check_ref_dtype(dt)
    if r is not None:
        if not x:
            return None
        if r is h5py.RegionReference or isinstance(x, h5py.RegionReference):
            return conv_region(x, ctx)
        return {"ref": ctx.f[x].name}
    if dt.kind == "V":
        return {"hex": bytes(x).hex()} if not isinstance(x, (bytes,)) else {"hex": x.hex()}
    if dt.kind == "b":
        return bool(x)
    if dt.kind in "iu":
        return int(x)
    if dt.kind == "f":
        return jfloat(x)
    if dt.kind == "c":
        return [jfloat(x.real), jfloat(x.imag)]
    if dt.kind == "S":
        return jbytes(x)
    if dt.kind == "U":
        return str(x)
    return repr(x)


def describe_selection(sid):
    out = {"dataspace": describe_space(sid)}
    st = sid.get_select_type()
    if st == h5py.h5s.SEL_HYPERSLABS:
        out["selection"] = "hyperslab"
        if sid.is_regular_hyperslab():
            start, stride, count, block = sid.get_regular_hyperslab()
            out["regular"] = {"start": [int(v) for v in start], "stride": [int(v) for v in stride],
                              "count": [int(v) for v in count], "block": [int(v) for v in block]}
        blocks = sid.get_select_hyper_blocklist()
        out["blocks"] = [{"start": [int(v) for v in b[0]], "end": [int(v) for v in b[1]]} for b in blocks]
    elif st == h5py.h5s.SEL_POINTS:
        out["selection"] = "points"
        out["points"] = [[int(v) for v in p] for p in sid.get_select_elem_pointlist()]
    elif st == h5py.h5s.SEL_ALL:
        out["selection"] = "all"
    else:
        out["selection"] = "none"
    out["npoints"] = int(sid.get_select_npoints())
    return out


def conv_region(ref, ctx):
    obj = ctx.f[ref]
    out = {"region_ref": obj.name}
    out.update(describe_selection(h5py.h5r.get_region(ref, ctx.f.id)))
    return out


def conv_any(arr, dt, ctx, array=False):
    if isinstance(arr, np.ndarray) and (arr.ndim > 0 or array):
        if arr.ndim == 0:
            return conv_value(arr[()], dt, ctx)
        return [conv_any(a, dt, ctx, array=True) if isinstance(a, np.ndarray) and a.ndim > 0
                else conv_value(a, dt, ctx) for a in arr]
    return conv_value(arr, dt, ctx)


def elem_dtype(dt):
    return dt.subdtype[0] if dt.subdtype is not None else dt


def summarize(arr, dt):
    flat = np.ascontiguousarray(arr).reshape(-1)
    s = {"count": int(flat.size)}
    if dt.kind in "iuf":
        le = flat.astype(flat.dtype.newbyteorder("<"))
        s["sha256_le"] = hashlib.sha256(le.tobytes()).hexdigest()
        s["sum"] = int(flat.astype(np.int64).sum()) if dt.kind in "iu" else float(flat.astype(np.float64).sum())
        s["min"] = int(flat.min()) if dt.kind in "iu" else jfloat(flat.min())
        s["max"] = int(flat.max()) if dt.kind in "iu" else jfloat(flat.max())
    cv = (lambda v: int(v)) if dt.kind in "iu" else (lambda v: jfloat(v)) if dt.kind == "f" else (lambda v: repr(v))
    s["first"] = [cv(v) for v in flat[:8]]
    s["last"] = [cv(v) for v in flat[-8:]]
    return s


def describe_data(raw, dt, ctx, count):
    if count <= SUMMARY_THRESHOLD:
        return {"data": conv_any(raw, elem_dtype(dt), ctx)}
    return {"data_summary": summarize(raw, elem_dtype(dt))}


def describe_space(sid):
    cls = sid.get_simple_extent_type()
    if cls == h5py.h5s.NULL:
        return {"class": "null"}
    if cls == h5py.h5s.SCALAR:
        return {"class": "scalar", "shape": []}
    mx = sid.get_simple_extent_dims(True)
    return {"class": "simple", "shape": [int(v) for v in sid.get_simple_extent_dims()],
            "maxshape": [None if v == h5py.h5s.UNLIMITED else int(v) for v in mx]}


def describe_attrs(obj, ctx):
    out = []
    for name in obj.attrs:  # h5py order: creation order if tracked, else name
        aid = h5py.h5a.open(obj.id, name.encode())
        a = {"name": name, "dtype": describe_type(aid.get_type()), "dataspace": describe_space(aid.get_space())}
        val = obj.attrs[name]
        if isinstance(val, h5py.Empty) or a["dataspace"]["class"] == "null":
            a["data"] = None
        else:
            raw = np.asarray(val) if not isinstance(val, (str, bytes)) else val
            if a["dataspace"]["class"] == "scalar":
                a["data"] = conv_value(raw[()] if isinstance(raw, np.ndarray) else raw, elem_dtype(aid.dtype), ctx) \
                    if aid.dtype.subdtype is None else conv_any(np.asarray(val), elem_dtype(aid.dtype), ctx, array=True)
            else:
                count = int(np.prod(aid.shape))
                a.update(describe_data(np.asarray(val), aid.dtype, ctx, count))
        out.append(a)
    return out


LAYOUTS = {h5py.h5d.COMPACT: "compact", h5py.h5d.CONTIGUOUS: "contiguous", h5py.h5d.CHUNKED: "chunked",
           h5py.h5d.VIRTUAL: "virtual"}
FILTER_NAMES = {1: "deflate", 2: "shuffle", 3: "fletcher32", 4: "szip", 5: "nbit", 6: "scaleoffset",
                32000: "lzf", 32001: "blosc", 307: "bzip2", 32004: "lz4", 32008: "bitshuffle", 32015: "zstd"}


def describe_dataset(ds, ctx):
    dcpl = ds.id.get_create_plist()
    sid = ds.id.get_space()
    d = {"kind": "dataset", "dtype": describe_type(ds.id.get_type()), "dataspace": describe_space(sid)}
    lay = dcpl.get_layout()
    d["layout"] = LAYOUTS.get(lay, str(lay))
    if lay == h5py.h5d.CHUNKED:
        d["chunk_shape"] = list(dcpl.get_chunk())
    filters = []
    for i in range(dcpl.get_nfilters()):
        code, flags, vals, name = dcpl.get_filter(i)
        filters.append({"id": code, "name": FILTER_NAMES.get(code, name.decode("utf-8", "replace")),
                        "flags": flags, "client_data": list(vals)})
    if filters:
        d["filters"] = filters
    d["storage_size"] = int(ds.id.get_storage_size())
    if lay in (h5py.h5d.CONTIGUOUS,):
        off = ds.id.get_offset()
        d["data_offset"] = off  # None if never allocated
    if lay == h5py.h5d.CHUNKED:
        n = ds.id.get_num_chunks()
        d["num_chunks_allocated"] = n
        if n <= 64:
            chunks = []
            for i in range(n):
                ci = ds.id.get_chunk_info(i)
                chunks.append({"offset": list(ci.chunk_offset), "filter_mask": ci.filter_mask,
                               "address": ci.byte_offset, "size": ci.size})
            d["chunks"] = chunks
    if lay == h5py.h5d.VIRTUAL:
        maps = []
        for i in range(dcpl.get_virtual_count()):
            vs = dcpl.get_virtual_vspace(i)
            ss = dcpl.get_virtual_srcspace(i)
            maps.append({"file": dcpl.get_virtual_filename(i), "dataset": dcpl.get_virtual_dsetname(i),
                         "virtual_selection": describe_selection(vs),
                         "source_selection": describe_selection(ss)})
        d["virtual_mappings"] = maps
    if dcpl.fill_value_defined() == h5py.h5d.FILL_VALUE_USER_DEFINED:
        try:
            fv = np.zeros((), dtype=ds.dtype)
            dcpl.get_fill_value(fv)
            d["fill_value"] = conv_value(fv[()], elem_dtype(ds.dtype), ctx)
        except Exception as e:  # e.g. vlen
            d["fill_value"] = "unavailable: %s" % e
    d["alloc_time"] = {h5py.h5d.ALLOC_TIME_EARLY: "early", h5py.h5d.ALLOC_TIME_LATE: "late",
                       h5py.h5d.ALLOC_TIME_INCR: "incremental"}.get(dcpl.get_alloc_time(), "default")
    if d["dataspace"]["class"] == "null":
        d["data"] = None
    else:
        count = int(np.prod(ds.shape))
        try:
            raw = ds[()]
        except OSError:
            # No conversion path in the reference library (e.g. bitfield):
            # read the raw element bytes with memory type == file type.
            ftype = ds.id.get_type()
            buf = np.zeros(ds.shape, dtype="V%d" % ftype.get_size())
            ds.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, mtype=ftype)
            d["data_encoding"] = "raw_hex"
            d["data"] = conv_any(buf, buf.dtype, ctx)
            d["attributes"] = describe_attrs(ds, ctx)
            return d
        if d["dataspace"]["class"] == "scalar":
            if ds.dtype.subdtype is None:
                d["data"] = conv_value(raw, ds.dtype, ctx)
            else:
                d["data"] = conv_any(np.asarray(raw), elem_dtype(ds.dtype), ctx, array=True)
        else:
            d.update(describe_data(raw, ds.dtype, ctx, count))
        ftype = ds.id.get_type()
        if ftype.get_class() == H5T.STRING and not ftype.is_variable_str() and count <= SUMMARY_THRESHOLD:
            # Exact stored element bytes (padding included), flattened row-major.
            buf = np.zeros(ds.shape, dtype="V%d" % ftype.get_size())
            ds.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, mtype=ftype)
            d["raw_elements_hex"] = [bytes(v).hex() for v in buf.reshape(-1)]
    d["attributes"] = describe_attrs(ds, ctx)
    return d


def describe_file(fname):
    objects = {}
    seen = {}
    with h5py.File(path(fname), "r") as f:
        ctx = Ctx(f)

        def visit(g, gpath):
            entry = {"kind": "group", "links": [], "attributes": describe_attrs(g, ctx)}
            objects[gpath] = entry
            for name in g:  # creation order if tracked, else name order
                lp = (gpath.rstrip("/") + "/" + name)
                lk = g.get(name, getlink=True)
                if isinstance(lk, h5py.SoftLink):
                    entry["links"].append({"name": name, "type": "soft", "target": lk.path})
                    continue
                if isinstance(lk, h5py.ExternalLink):
                    entry["links"].append({"name": name, "type": "external", "file": lk.filename, "target": lk.path})
                    continue
                obj = g[name]
                key = hash(obj.id)
                link = {"name": name, "type": "hard"}
                if key in seen:
                    link["target"] = seen[key]
                    entry["links"].append(link)
                    continue
                seen[key] = lp
                link["target"] = lp
                entry["links"].append(link)
                if isinstance(obj, h5py.Group):
                    visit(obj, lp)
                elif isinstance(obj, h5py.Dataset):
                    objects[lp] = describe_dataset(obj, ctx)
                elif isinstance(obj, h5py.Datatype):
                    objects[lp] = {"kind": "datatype", "dtype": describe_type(obj.id),
                                   "attributes": describe_attrs(obj, ctx)}

        seen[hash(f.id)] = "/"
        visit(f, "/")
    for p, extra in EXPECT.get(fname, {}).items():
        objects[p].setdefault("expect", {}).update(extra)
    return objects


def superblock_info(fname):
    with open(path(fname), "rb") as fh:
        data = fh.read()
    sig = b"\x89HDF\r\n\x1a\n"
    off = 0
    while off < len(data):
        if data[off:off + 8] == sig:
            return off, data[off + 8]
        off = 512 if off == 0 else off * 2
    raise RuntimeError("no superblock in %s" % fname)


def sanity_check(fname, objects):
    """Black-box check that the expected chunk index structures exist."""
    with open(path(fname), "rb") as fh:
        data = fh.read()
    sigs = {"fixed_array": b"FAHD", "extensible_array": b"EAHD", "btree_v2": b"BTHD", "btree_v1": b"TREE"}
    for p, o in objects.items():
        ci = o.get("expect", {}).get("chunk_index")
        if ci in sigs and sigs[ci] not in data:
            raise RuntimeError("%s:%s expected %s but no %r signature" % (fname, p, ci, sigs[ci]))


def dump_json(obj):
    """Indented JSON; flat lists and small leaf objects are kept on one line."""
    compact = []

    def is_flat(v):
        if isinstance(v, dict):
            return False
        if isinstance(v, list):
            return not any(isinstance(x, dict) for x in json_leaves(v))
        return True

    def walk(o):
        if isinstance(o, dict):
            if len(o) <= 8 and all(is_flat(v) for v in o.values()):
                compact.append(json.dumps(o, ensure_ascii=False, separators=(", ", ": ")))
                return "@@COMPACT%d@@" % (len(compact) - 1)
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            if any(isinstance(x, dict) for x in json_leaves(o)):
                return [walk(x) for x in o]
            compact.append(json.dumps(o, ensure_ascii=False, separators=(",", ":")))
            return "@@COMPACT%d@@" % (len(compact) - 1)
        return o

    text = json.dumps(walk(obj), indent=1, ensure_ascii=False)
    for i in range(len(compact) - 1, -1, -1):
        text = text.replace('"@@COMPACT%d@@"' % i, compact[i])
    return text


def json_leaves(lst):
    for x in lst:
        if isinstance(x, list):
            yield from json_leaves(x)
        else:
            yield x


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures"))
    args = ap.parse_args()
    OUT = os.path.abspath(args.out)
    os.makedirs(OUT, exist_ok=True)
    for fn in os.listdir(OUT):
        if fn.endswith(".h5"):
            os.remove(os.path.join(OUT, fn))

    gen_superblocks()
    gen_groups()
    gen_dtypes()
    gen_dataspaces()
    gen_layouts()
    gen_filters()
    gen_attributes()
    gen_vds()
    gen_misc()

    files = {}
    for fname in sorted(FILE_NOTES):
        objs = describe_file(fname)
        sanity_check(fname, objs)
        sb_off, sb_ver = superblock_info(fname)
        entry = dict(FILE_NOTES[fname])
        with open(path(fname), "rb") as fh:
            content = fh.read()
        entry.update({"size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                      "superblock_offset": sb_off, "superblock_version": sb_ver, "objects": objs})
        files[fname] = entry

    manifest = {
        "schema_version": 1,
        "generator": "tools/gen_fixtures.py",
        "h5py_version": h5py.version.version,
        "hdf5_version": h5py.version.hdf5_version,
        "numpy_version": np.__version__,
        "hdf5plugin_version": getattr(hdf5plugin, "version", None) if hdf5plugin else None,
        "files": files,
    }
    with open(path("manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(dump_json(manifest))
        fh.write("\n")
    with open(path("files.txt"), "w") as fh:
        for fname in sorted(files):
            fh.write("%s %d\n" % (fname, files[fname]["superblock_offset"]))
    total = sum(e["size"] for e in files.values())
    for fname in sorted(files):
        print("%-28s %8d bytes  sb v%d" % (fname, files[fname]["size"], files[fname]["superblock_version"]))
    print("total: %d bytes in %d files" % (total, len(files)), file=sys.stderr)


if __name__ == "__main__":
    main()
