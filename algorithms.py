import os, glob
import numpy as np
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QVariant as QV
from qgis.core import (
    Qgis,
    QgsAbstractGeometryTransformer as GT,
    QgsApplication,
    QgsCoordinateReferenceSystem as CRS,
    QgsCoordinateTransform as CT,
    QgsCsException,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry as G,
    QgsLineString,
    QgsPointXY as P,
    QgsProcessing as PR,
    QgsProcessingAlgorithm,
    QgsProcessingException as E,
    QgsProcessingParameterBoolean as Bo,
    QgsProcessingParameterEnum as En,
    QgsProcessingParameterFeatureSink as Sk,
    QgsProcessingParameterFeatureSource as Src,
    QgsProcessingParameterMapLayer as Ly,
    QgsProcessingParameterNumber as Nm,
    QgsProcessingParameterRasterDestination as RD,
    QgsProcessingParameterString as St,
    QgsPoint,
    QgsRasterFileWriter,
    QgsRasterLayer,
    QgsRectangle,
    QgsWkbTypes as T,
)
from .mesh import ProjectionMesh

D = os.path.dirname(__file__)
ICON_PATH = os.path.join(D, "icon.png")
Dbl = Nm.Double
CRSS = (
    ("Use the layer's CRS", None),
    ("Equirectangular longitude/latitude (EPSG:4326)", "EPSG:4326"),
)


def user_folder():
    return os.path.join(QgsApplication.qgisSettingsDirPath(), "geobica_projections")


def available_projections():
    return dict(
        sorted(
            {
                os.path.splitext(os.path.basename(p))[0]: p
                for d in (os.path.join(D, "projections"), user_folder())
                for p in sorted(glob.glob(os.path.join(d, "*.npz")))
            }.items()
        )
    )


try:
    from qgis.PyQt.QtWidgets import QCheckBox, QComboBox
    from processing.gui.wrappers import WidgetWrapper as WW, DIALOG_STANDARD

    class DependentCheck(WW):
        def __init__(s, param, dialog, row=0, col=0, parent=None, **k):
            s.parent_name = parent
            super().__init__(param, dialog, row, col, **k)

        def _std(s):
            return s.dialogType == DIALOG_STANDARD

        def createLabel(s):
            return None if s._std() else super().createLabel()

        def createWidget(s, **k):
            if s._std():
                return QCheckBox()
            w = QComboBox()
            w.addItem(s.tr("Yes"), True)
            w.addItem(s.tr("No"), False)
            return w

        def setValue(s, v):
            if v is None:
                return
            if s._std():
                s.widget.setChecked(bool(v))
            else:
                s.widget.setCurrentIndex(0 if v else 1)

        def value(s):
            return s.widget.isChecked() if s._std() else s.widget.currentData()

        def postInitialize(s, ws):
            if not s._std():
                return
            for w in ws:
                if w.parameterDefinition().name() == s.parent_name:
                    f = lambda *_, w=w: s.widget.setEnabled(bool(w.parameterValue()))
                    w.widgetValueHasChanged.connect(f)
                    f()

except ImportError:
    DependentCheck = None
_c = {}


def load_projection(p):
    k = (p, os.path.getmtime(p))
    if k not in _c:
        _c.clear()
        _c[k] = ProjectionMesh(p)
    return _c[k]


class _C(GT):
    def __init__(s):
        super().__init__()
        s.x = []
        s.y = []

    def transformPoint(s, x, y, z, m):
        s.x.append(x)
        s.y.append(y)
        return True, x, y, z, m


class _R(GT):
    def __init__(s, x, y):
        super().__init__()
        s.p = zip(x, y).__next__

    def transformPoint(s, x, y, z, m):
        a, b = s.p()
        return True, a, b, z, m


def map_vertices(gs, fn):
    c = _C()
    for g in gs:
        g.get().transform(c)
    if c.x:
        r = _R(*(v.tolist() for v in fn(np.array(c.x), np.array(c.y))))
        for g in gs:
            g.get().transform(r)


def _keep(g, t):
    if g.isEmpty() or T.flatType(g.wkbType()) != T.GeometryCollection:
        return g
    p = [G(q.clone()) for q in g.constGet() if T.geometryType(q.wkbType()) == t]
    return G.collectGeometry(p) if p else G()


def fields(*f):
    r = QgsFields()
    for n, t in f:
        r.append(QgsField(n, t))
    return r


def source_crs(a, p, c, s):
    k = CRSS[a.parameterAsEnum(p, "SOURCE_CRS", c)][1]
    r = CRS(k) if k else (s.sourceCrs() if hasattr(s, "sourceCrs") else s.crs())
    if not r.isValid():
        raise E(
            'The input layer has no CRS. Set "Treat input coordinates as" to equirectangular longitude/latitude.'
        )
    return r


def Z(sc, x, y):
    z = (np.asarray(x) + 1j * np.asarray(y)) * sc
    return z.real, z.imag


NP = {
    getattr(Qgis.DataType, k): np.dtype(v)
    for k, v in (
        ("Byte", "u1"),
        ("Int8", "i1"),
        ("UInt16", "u2"),
        ("Int16", "i2"),
        ("UInt32", "u4"),
        ("Int32", "i4"),
        ("Float32", "f4"),
        ("Float64", "f8"),
    )
    if hasattr(Qgis.DataType, k)
}


def transform_xy(ct, x, y):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    o = np.full((2, x.size), np.nan)
    f = np.isfinite(x) & np.isfinite(y)
    i = np.where(f)[0]

    def go(a, b):
        try:
            g = QgsLineString(x[i[a:b]].tolist(), y[i[a:b]].tolist())
            g.transform(ct)
            w = bytes(g.asWkb())
            o[:, i[a:b]] = (
                np.frombuffer(w, "<>"[w[0] == 0] + "f8", offset=9).reshape(-1, 2).T
            )
        except QgsCsException:
            if b - a > 1:
                k = (a + b) // 2
                go(a, k)
                go(k, b)

    for a in range(0, i.size, 1000000):
        go(a, min(i.size, a + 1000000))
    o[~np.isfinite(o)] = np.nan
    return o


def grid_inside(B, x0, y0, d, nc, nr):
    a, b = B[:-1], B[1:]
    ra, rb = (y0 - a.imag) / d, (y0 - b.imag) / d
    lo, hi = (
        np.clip(np.ceil(f(ra, rb)), 0, nr).astype(np.int64)
        for f in (np.minimum, np.maximum)
    )
    n = hi - lo
    e = np.repeat(np.arange(a.size), n)
    r = np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    x = a.real[e] + (r - ra[e]) / (rb[e] - ra[e]) * (b.real[e] - a.real[e])
    t = np.zeros((nr, nc + 1), np.uint8)
    np.add.at(t, (r, np.clip(np.ceil((x - x0) / d), 0, nc).astype(np.int64)), 1)
    return (np.cumsum(t, 1, dtype=np.uint8)[:, :nc] & 1).astype(bool)


def engine(g):
    e = G.createGeometryEngine(g.constGet())
    e.prepareGeometry()
    return e


class _A(QgsProcessingAlgorithm):
    def icon(s):
        return QIcon(ICON_PATH)

    def group(s):
        return ""

    def groupId(s):
        return ""

    def createInstance(s):
        return type(s)()

    def name(s):
        return s.N[0]

    def displayName(s):
        return s.N[1]

    def shortHelpString(s):
        return s.N[2]

    def _proj(s):
        s._p = available_projections()
        n = list(s._p)
        s.addParameter(
            En(
                "PROJECTION",
                "Computed projection",
                options=n,
                defaultValue=n.index("world") if "world" in n else 0 if n else None,
            )
        )
        s.addParameter(
            Nm(
                "ROTATION",
                "Rotation (degrees, counter-clockwise)",
                type=Dbl,
                defaultValue=0.0,
            )
        )

    def _mesh(s, p, c):
        n = list(s._p)
        if not n:
            raise E(
                f"No projections were found in {user_folder()}. Did you delete them?"
            )
        m = load_projection(s._p[n[s.parameterAsEnum(p, "PROJECTION", c)]])
        return m, m.meters_per_unit * np.exp(
            1j * np.radians(s.parameterAsDouble(p, "ROTATION", c))
        )

    def _clip(s, m):
        x, y = m.boundary_w()
        g = G.fromPolygonXY([[P(*v) for v in zip(x, y)]]).makeValid()
        return g, engine(g)

    def _src(s, p, c):
        r = s.parameterAsSource(p, "INPUT", c)
        if r is None:
            raise E(s.invalidSourceError(p, "INPUT"))
        return r


class ProjectLayerAlgorithm(_A):
    N = (
        "project_layer",
        "Apply optimized conformal projection",
        "Reprojects a vector or raster layer into a precomputed minimum distortion conformal projection. The output will be projectionless, with units in meters along the boundary. Only shapes within the boundary polygon will be projected.",
    )

    def initAlgorithm(s, config=None):
        s.addParameter(
            Ly("INPUT", "Input layer", types=[PR.TypeVectorAnyGeometry, PR.TypeRaster])
        )
        s._proj()
        s.addParameter(
            En(
                "SOURCE_CRS",
                "Treat input coordinates as",
                options=[l for l, _ in CRSS],
                defaultValue=0,
            )
        )
        s.addParameter(
            Nm(
                "DENSIFY",
                "Densify (Add vertices along edges every x units before projecting, 0 to disable)",
                type=Dbl,
                minValue=0.0,
                defaultValue=0.1,
            )
        )
        s.addParameter(
            Nm(
                "PIXEL_SIZE",
                "Raster pixel size (meters, 0 for automatic)",
                type=Dbl,
                minValue=0.0,
                defaultValue=0.0,
            )
        )
        s.addParameter(
            En(
                "RESAMPLING",
                "Raster resampling",
                options=["Nearest neighbor", "Bilinear"],
                defaultValue=0,
            )
        )
        s.addParameter(
            Sk(
                "OUTPUT",
                "Projected layer (vector input)",
                PR.TypeVectorAnyGeometry,
                None,
                True,
            )
        )
        s.addParameter(
            RD("OUTPUT_RASTER", "Projected raster (raster input)", None, True)
        )

    def processAlgorithm(s, p, c, fb):
        l = s.parameterAsLayer(p, "INPUT", c)
        if isinstance(l, QgsRasterLayer):
            return s._raster(l, p, c, fb)
        src = s._src(p, c)
        m, sc = s._mesh(p, c)
        dd = s.parameterAsDouble(p, "DENSIFY", c)
        cr = source_crs(s, p, c, src)
        tl = CT(cr, CRS("EPSG:4326"), c.transformContext())
        ds = dd if cr.isGeographic() else dd * 111319.49079327357
        gt = T.geometryType(src.wkbType())
        ot = src.wkbType()
        if gt in (T.LineGeometry, T.PolygonGeometry):
            ot = T.multiType(ot)
        fs = QgsFields(src.fields())
        sk, did = s.parameterAsSink(p, "OUTPUT", c, fs, ot, CRS())
        if sk is None:
            raise E(s.invalidSinkError(p, "OUTPUT"))
        cg, ce = s._clip(m)
        fb.pushInfo(
            f"Projection: {m.name} (centered on {m.center_lat:.3f}, {m.center_lon:.3f})"
        )
        n = dict.fromkeys(
            ("written", "outside", "failed", "vertices", "extrapolated"), 0
        )

        def tm(x, y):
            a, b, e = m.w_to_m(x, y)
            n["vertices"] += x.size
            n["extrapolated"] += int(e.sum())
            return Z(sc, a, b)

        def flush(b):
            if not b:
                return
            map_vertices([g for _, g in b], m.lonlat_to_w)
            k = []
            for f, g in b:
                if not ce.intersects(g.constGet()):
                    n["outside"] += 1
                    continue
                if not ce.contains(g.constGet()):
                    h = _keep(g.intersection(cg), gt)
                    if h.isEmpty():
                        h = _keep(_keep(g.makeValid(), gt).intersection(cg), gt)
                    if h.isEmpty():
                        n["outside"] += 1
                        continue
                    g = h
                k.append((f, g))
            map_vertices([g for _, g in k], tm)
            for f, g in k:
                if T.isMultiType(ot):
                    g.convertToMultiType()
                o = QgsFeature(fs)
                o.setGeometry(g)
                o.setAttributes(f.attributes())
                sk.addFeature(o, QgsFeatureSink.FastInsert)
                n["written"] += 1

        tot = src.featureCount() or 1
        b = []
        v = 0
        for i, f in enumerate(src.getFeatures()):
            if fb.isCanceled():
                break
            g = G(f.geometry())
            if g.isNull() or g.isEmpty():
                n["outside"] += 1
                continue
            if ds > 0 and gt != T.PointGeometry:
                g = g.densifyByDistance(ds)
            try:
                g.transform(tl)
            except QgsCsException:
                n["failed"] += 1
                continue
            b.append((f, g))
            v += g.constGet().nCoordinates()
            if len(b) >= 500 or v >= 500000:
                flush(b)
                b = []
                v = 0
                fb.setProgress(100 * (i + 1) / tot)
        flush(b)
        fb.pushInfo(
            f"{n['written']} features written, {n['outside']} empty or outside the boundary, {n['failed']} could not be converted to longitude/latitude."
        )
        if n["vertices"]:
            fb.pushInfo(
                f"{100*n['extrapolated']/n['vertices']:.2f}% of vertices lie outside the sample mesh and were extrapolated using the nearest triangle."
            )
        return {"OUTPUT": did}

    def _raster(s, l, p, c, fb):
        from osgeo import gdal

        o = s.parameterAsOutputLayer(p, "OUTPUT_RASTER", c)
        if not o:
            raise E("Choose an output file for the projected raster.")
        m, sc = s._mesh(p, c)
        cr = source_crs(s, p, c, l)
        pr = l.dataProvider()
        ex = l.extent()
        W0, H0 = l.width(), l.height()
        vr = not (W0 and H0)
        if vr:
            a = ex.height() / ex.width()
            W0, H0 = (
                (4096, max(1, round(4096 * a)))
                if a <= 1
                else (max(1, round(4096 / a)), 4096)
            )
        rx, ry = ex.width() / W0, ex.height() / H0
        X0, Y0 = ex.xMinimum(), ex.yMaximum()
        ll = CRS("EPSG:4326")
        fw, bw = CT(cr, ll, c.transformContext()), CT(ll, cr, c.transformContext())
        geo = cr.isGeographic()
        bil = s.parameterAsEnum(p, "RESAMPLING", c) == 1
        fb.pushInfo(
            f"Projection: {m.name} (centered on {m.center_lat:.3f}, {m.center_lon:.3f})"
        )
        cg, ce = s._clip(m)
        B = m.boundary_mapped() * sc
        B = np.append(B, B[:1])
        sx, sy = (np.linspace(0.5, n - 0.5, min(n, 257)) for n in (W0, H0))
        C, Rw = np.meshgrid(sx, sy)
        lo, la = transform_xy(fw, X0 + C * rx, Y0 - Rw * ry)
        Q = np.full(lo.size, np.nan + 0j)
        i = np.where(np.isfinite(lo))[0]
        if i.size:
            wx, wy = m.lonlat_to_w(lo[i], la[i])
            i = i[
                [ce.contains(QgsPoint(a, b)) for a, b in zip(wx.tolist(), wy.tolist())]
            ]
            if i.size:
                Q[i] = complex(sc) * (lambda a, b, _: a + 1j * b)(
                    *m.w_to_m(*m.lonlat_to_w(lo[i], la[i]))
                )
        if not i.size:
            raise E("Raster is outside the projection boundary.")
        Q = Q.reshape(C.shape)
        r = np.concatenate(
            [
                (np.abs(np.diff(Q, axis=k)) / np.expand_dims(np.diff(v), 1 - k)).ravel()
                for k, v in ((1, sx), (0, sy))
                if Q.shape[k] > 1
            ]
        )
        r = r[np.isfinite(r) & (r > 0)]
        hi = float(np.percentile(r, 99)) if r.size else 0.0
        px = s.parameterAsDouble(p, "PIXEL_SIZE", c)
        q = Q[np.isfinite(Q)]
        pd = 2 * max(W0 / sx.size, H0 / sy.size) * hi
        bx = [
            max(q.real.min() - pd, B.real.min()),
            max(q.imag.min() - pd, B.imag.min()),
            min(q.real.max() + pd, B.real.max()),
            min(q.imag.max() + pd, B.imag.max()),
        ]
        if px <= 0:
            px = (
                float(
                    f"{max(bx[2]-bx[0],bx[3]-bx[1])/4000 if vr else np.median(r):.3g}"
                )
                if r.size
                else max(bx[2] - bx[0], bx[3] - bx[1]) / 1000
            )
            k = (bx[2] - bx[0]) * (bx[3] - bx[1]) / px / px
            if k > 1e8:
                px = float(f"{px*(k/1e8)**.5:.3g}")
                fb.pushInfo(
                    "The automatic pixel size was increased to keep the raster under 100 million pixels."
                )
        x0, y1 = np.floor(bx[0] / px) * px, np.ceil(bx[3] / px) * px
        W, H = int(max(1, np.ceil((bx[2] - x0) / px))), int(
            max(1, np.ceil((y1 - bx[1]) / px))
        )
        f = max(1, int(px / hi)) if hi > 0 and not vr else 1
        if vr and hi > 0:
            k = 2 * hi / px
            W0, H0 = max(1, int(np.ceil(W0 * k))), max(1, int(np.ceil(H0 * k)))
            rx, ry = ex.width() / W0, ex.height() / H0
        ty = [pr.dataType(b) for b in range(1, l.bandCount() + 1)]
        argb = ty[0] in (Qgis.DataType.ARGB32, Qgis.DataType.ARGB32_Premultiplied)
        if argb:
            dt, nd, al = np.dtype("u1"), [None] * 4, 3
        else:
            if any(t not in NP for t in ty):
                raise E("This raster data type is not supported.")
            dt = np.result_type(*(NP[t] for t in ty))
            nd = [
                (
                    pr.sourceNoDataValue(b)
                    if pr.sourceHasNoDataValue(b) and pr.useSourceNoDataValue(b)
                    else None
                )
                for b in range(1, len(ty) + 1)
            ]
            al = None
            if dt == np.uint8 and len(ty) in (3, 4) and all(v is None for v in nd):
                al = 3
                nd = [None] * 4
            else:
                nd = [
                    (
                        v
                        if v is not None
                        else (
                            np.nan
                            if dt.kind == "f"
                            else (
                                np.iinfo(dt).min if dt.kind == "i" else np.iinfo(dt).max
                            )
                        )
                    )
                    for v in nd
                ]
        nc = (
            al is not None
            and QgsRasterFileWriter.driverForExtension(os.path.splitext(o)[1])
            == "netCDF"
        )
        if nc:
            nd, al = [0, 0, 0], None
        nb = len(nd)
        fb.pushInfo(
            f'Output: {W} x {H} pixels of {px:g} m, {nb} band{"s"*(nb>1)}'
            + (f", reading the source at 1/{f} resolution" if f > 1 else "")
        )

        def F(x, y):
            a, b, _ = m.m_to_w(*Z(1 / sc, x, y))
            X, Y = transform_xy(bw, *m.w_to_lonlat(a, b))
            if geo:
                X = X0 + np.mod(X - X0, 360)
            return (X - X0) / rx, (Y0 - Y) / ry

        def read(c0, r0, c1, r1):
            nw, nh = -(-(c1 - c0) // f), -(-(r1 - r0) // f)
            e = QgsRectangle(X0 + c0 * rx, Y0 - r1 * ry, X0 + c1 * rx, Y0 - r0 * ry)
            A = []
            for b, t in enumerate(ty, 1):
                k = pr.block(b, e, nw, nh)
                a = np.frombuffer(bytes(k.data()), "u4" if argb else NP[t])
                del k
                if a.size != nw * nh:
                    raise E(f"Could not read band {b} of the raster.")
                a = a.reshape(nh, nw)
                A += (
                    [a.view("u1").reshape(nh, nw, 4)[..., k] for k in (2, 1, 0, 3)]
                    if argb
                    else [a]
                )
            return A, nw, nh, (c1 - c0) / nw, (r1 - r0) / nh

        def ok(a, v):
            return (
                np.ones(a.shape, bool)
                if v is None
                else ~np.isnan(a) if v != v else a != v
            )

        def sample(a, v, u, w, nw, nh):
            j, i = np.clip(u.astype(np.int64), 0, nw - 1), np.clip(
                w.astype(np.int64), 0, nh - 1
            )
            z = a[i, j]
            if not bil:
                return z
            x, y = u - 0.5, w - 0.5
            j, i = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            fx, fy = x - j, y - i
            j0, j1, i0, i1 = (
                np.clip(j, 0, nw - 1),
                np.clip(j + 1, 0, nw - 1),
                np.clip(i, 0, nh - 1),
                np.clip(i + 1, 0, nh - 1),
            )
            P = [a[i0, j0], a[i0, j1], a[i1, j0], a[i1, j1]]
            g = ok(P[0], v) & ok(P[1], v) & ok(P[2], v) & ok(P[3], v)
            P = [t.astype(float) for t in P]
            y = (P[0] * (1 - fx) + P[1] * fx) * (1 - fy) + (
                P[2] * (1 - fx) + P[3] * fx
            ) * fy
            if dt.kind != "f":
                y = np.clip(np.round(y), *(lambda k: (k.min, k.max))(np.iinfo(dt)))
            return np.where(g, y, z).astype(dt)

        dr = QgsRasterFileWriter.driverForExtension(os.path.splitext(o)[1]) or "GTiff"
        mk = dr == "GTiff"
        op = (
            ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
            if dr == "GTiff" or not mk
            else []
        )
        tp = o if mk else o + ".tmp.tif"
        ds = gdal.GetDriverByName("GTiff" if not mk else dr).Create(
            tp,
            W,
            H,
            nb,
            gdal.GetDataTypeByName(
                {
                    "u1": "Byte",
                    "i1": "Int8",
                    "u2": "UInt16",
                    "i2": "Int16",
                    "u4": "UInt32",
                    "i4": "Int32",
                    "f4": "Float32",
                    "f8": "Float64",
                }[dt.str[1:]]
            ),
            op,
        )
        if ds is None:
            raise E(f"Could not create {o}.")
        ds.SetGeoTransform([x0, px, 0, y1, 0, -px])
        bd = [ds.GetRasterBand(k + 1) for k in range(nb)]
        for k, v in enumerate(nd):
            if v is not None:
                bd[k].SetNoDataValue(float(v))
        if al is not None:
            for k, g in enumerate(
                (
                    gdal.GCI_RedBand,
                    gdal.GCI_GreenBand,
                    gdal.GCI_BlueBand,
                    gdal.GCI_AlphaBand,
                )
            ):
                bd[k].SetColorInterpretation(g)
        S = 16
        L = -(-W // S)
        rs = S * max(1, int(4e6 / (W * S)))
        tl = 0.125 * f
        n = 0
        for i0 in range(0, H, rs):
            if fb.isCanceled():
                break
            R_ = min(rs, H - i0)
            K = -(-R_ // S)
            yt = y1 - (i0 + 0.5) * px
            out = [np.full((R_, W), 0 if v is None else v, dt) for v in nd]
            mask = grid_inside(B, x0 + 0.5 * px, yt, px, W, R_)
            if mask.any():

                def ex_(msk, xa, ya):
                    u, v = np.full(msk.shape, np.nan), np.full(msk.shape, np.nan)
                    if msk.any():
                        u[msk], v[msk] = F(xa[msk], ya[msk])
                    return u, v

                jj, kk = np.meshgrid(np.arange(L + 1), np.arange(K + 1))
                Nu, Nv = ex_(
                    grid_inside(B, x0 + 0.5 * px, yt, S * px, L + 1, K + 1),
                    x0 + (jj * S + 0.5) * px,
                    yt - kk * S * px,
                )
                jj, kk = np.meshgrid(np.arange(L), np.arange(K))
                Cu, Cv = ex_(
                    grid_inside(
                        B, x0 + (0.5 + S / 2) * px, yt - S / 2 * px, S * px, L, K
                    ),
                    x0 + (jj * S + 0.5 + S / 2) * px,
                    yt - (kk * S + S / 2) * px,
                )
                av = lambda N: (N[:-1, :-1] + N[:-1, 1:] + N[1:, :-1] + N[1:, 1:]) / 4
                g = (np.abs(av(Nu) - Cu) <= tl) & (np.abs(av(Nv) - Cv) <= tl)
                ri, ci = np.nonzero(mask)
                k, j = ri // S, ci // S
                fy, fx = (ri % S) / S, (ci % S) / S
                h = g[k, j]
                u, v = np.full(ri.size, np.nan), np.full(ri.size, np.nan)
                for U, N in ((u, Nu), (v, Nv)):
                    U[h] = (
                        (N[k, j] * (1 - fx) + N[k, j + 1] * fx) * (1 - fy)
                        + (N[k + 1, j] * (1 - fx) + N[k + 1, j + 1] * fx) * fy
                    )[h]
                e = ~h
                if e.any():
                    u[e], v[e] = F(x0 + (ci[e] + 0.5) * px, yt - ri[e] * px)
                z = (
                    np.isfinite(u)
                    & np.isfinite(v)
                    & (u >= 0)
                    & (u < W0)
                    & (v >= 0)
                    & (v < H0)
                )
                ri, ci, u, v = ri[z], ci[z], u[z], v[z]
                if ri.size:
                    c0, r0 = max(0, int(u.min()) - 1), max(0, int(v.min()) - 1)
                    c0 -= c0 % f
                    r0 -= r0 % f
                    A, nw, nh, kx, ky = read(
                        c0, r0, min(W0, int(u.max()) + 2), min(H0, int(v.max()) + 2)
                    )
                    u, v = (u - c0) / kx, (v - r0) / ky
                    for b, a in enumerate(A[:nb]):
                        out[b][ri, ci] = (
                            np.maximum(sample(a, None, u, v, nw, nh), 1)
                            if nc
                            else sample(a, nd[b], u, v, nw, nh)
                        )
                    if al is not None and len(A) == 3:
                        out[3][ri, ci] = 255
                    if nc and len(A) > 3:
                        t = sample(A[3], None, u, v, nw, nh) == 0
                        for b in range(3):
                            out[b][ri[t], ci[t]] = 0
                    n += ri.size
            for b in range(nb):
                bd[b].WriteRaster(0, i0, W, R_, np.ascontiguousarray(out[b]).tobytes())
            fb.setProgress(100 * (i0 + R_) / H)
        if dr == "netCDF" and nb > 1:
            ds.SetMetadata(
                {
                    "NETCDF_DIM_EXTRA": "{band}",
                    "NETCDF_DIM_band_DEF": f"{{{nb},6}}",
                    "NETCDF_DIM_band_VALUES": "{"
                    + ",".join(map(str, range(1, nb + 1)))
                    + "}",
                }
            )
            for k, b in enumerate(bd):
                b.SetMetadata(
                    {"NETCDF_VARNAME": "Band1", "NETCDF_DIM_band": str(k + 1)}
                )
        bd = None
        ds.FlushCache()
        ds = None
        if not mk:
            v = gdal.GetDriverByName(dr)
            v = v and v.CreateCopy(
                o,
                gdal.Open(tp),
                options=["FORMAT=NC4", "COMPRESS=DEFLATE"] if dr == "netCDF" else [],
            )
            gdal.Unlink(tp)
            v = None
            q = QgsRasterLayer(
                o,
                "",
            )
            if not q.isValid() or q.bandCount() != nb:
                raise E(
                    f"The raster could not be written as {dr}. Use a .tif output file."
                )
        fb.pushInfo(f"{n} pixels written inside the boundary.")
        return {"OUTPUT_RASTER": o}


class BoundaryGraticuleAlgorithm(_A):
    N = (
        "boundary_and_statistics",
        "Projection boundary and statistics",
        "Produces the boundary polygon (M) after projection, a graticule, and isolines of distortion (as well as points of minimum scale).",
    )

    def initAlgorithm(s, config=None):
        s._proj()
        s.addParameter(
            Nm(
                "STEP",
                "Graticule spacing (degrees)",
                type=Dbl,
                minValue=0.01,
                defaultValue=5.0,
            )
        )
        s.addParameter(Sk("BOUNDARY", "Boundary", PR.TypeVectorPolygon))
        s.addParameter(Sk("GRATICULE", "Graticule", PR.TypeVectorLine))
        s.addParameter(
            St(
                "RATIOS",
                "Included isolines of distortion (comma-separated)",
                defaultValue="0.99, 0.98, 0.95, 0.9, 0.8, 0.5",
            )
        )
        s.addParameter(
            Bo(
                "CENTER",
                "Designate 1.0 as the midpoint rather than boundary scale",
                defaultValue=False,
            )
        )
        q = Bo("LOG_CENTER", "Define midpoint as geometric mean", defaultValue=False)
        if DependentCheck:
            q.setMetadata(
                {"widget_wrapper": {"class": DependentCheck, "parent": "CENTER"}}
            )
        s.addParameter(q)
        s.addParameter(
            Sk("DISTORTION", "Isolines of distortion", PR.TypeVectorLine, optional=True)
        )
        s.addParameter(
            Sk(
                "MINIMUMS",
                "Local minimums of distortion",
                PR.TypeVectorPoint,
                optional=True,
            )
        )

    def processAlgorithm(s, p, c, fb):
        m, sc = s._mesh(p, c)
        st = s.parameterAsDouble(p, "STEP", c)
        sink = lambda k, f, t: s.parameterAsSink(p, k, c, f, t, CRS())
        f = fields(("projection", QV.String))
        sk, bid = sink("BOUNDARY", f, T.Polygon)
        o = QgsFeature(f)
        o.setGeometry(
            G.fromPolygonXY([[P(*Z(sc, v.real, v.imag)) for v in m.boundary_mapped()]])
        )
        o.setAttributes([m.name])
        sk.addFeature(o, QgsFeatureSink.FastInsert)
        f = fields(("kind", QV.String), ("degrees", QV.Double))
        sk, gid = sink("GRATICULE", f, T.MultiLineString)
        la, lo = np.arange(-90, 90.05, 0.1), np.arange(-180, 180.05, 0.1)
        L = [
            ("meridian", float(x), np.full_like(la, x), la)
            for x in np.arange(-180, 180, st)
        ] + [
            ("parallel", float(y), lo, np.full_like(lo, y))
            for y in np.arange(-90 + st, 90, st)
        ]
        cg, ce = s._clip(m)
        for i, (k, v, x, y) in enumerate(L):
            if fb.isCanceled():
                break
            a, b = m.lonlat_to_w(x, y)
            g = G(QgsLineString(a.tolist(), b.tolist()))
            if not ce.intersects(g.constGet()):
                continue
            g = _keep(g.intersection(cg), T.LineGeometry)
            if g.isEmpty():
                continue
            map_vertices([g], lambda x, y: Z(sc, *m.w_to_m(x, y)[:2]))
            g.convertToMultiType()
            o = QgsFeature(f)
            o.setGeometry(g)
            o.setAttributes([k, v])
            sk.addFeature(o, QgsFeatureSink.FastInsert)
            fb.setProgress(100 * (i + 1) / len(L))
        r = {"BOUNDARY": bid, "GRATICULE": gid}
        rf = 1.0
        if s.parameterAsBool(p, "CENTER", c):
            ms = m.minimum_scale_factor
            lm = (
                float(ms.min())
                if ms is not None and ms.size
                else float(np.exp(np.nanmin(m.nodal_log_scale())))
            )
            rf = lm**0.5 if s.parameterAsBool(p, "LOG_CENTER", c) else (1 + lm) / 2
            fb.pushInfo(
                f"Distortion is measured from {rf:.5f} times the boundary scale: the boundary is {1/rf:.4f} and the global minimum is {lm/rf:.4f}."
            )
        f = fields(("area_factor", QV.Double), ("scale_factor", QV.Double))
        sk, did = sink("DISTORTION", f, T.LineString)
        if sk is not None:
            try:
                F = sorted(
                    {
                        float(v)
                        for v in s.parameterAsString(p, "RATIOS", c)
                        .replace(";", ",")
                        .split(",")
                        if v.strip()
                    }
                )
            except ValueError:
                raise E("Scale factors must be numbers separated by commas.")
            for a in F:
                if fb.isCanceled():
                    break
                for w in m.isolines(np.log(a * rf)):
                    x, y = Z(sc, *m.w_to_m(w.real, w.imag)[:2])
                    o = QgsFeature(f)
                    o.setGeometry(G.fromPolylineXY([P(u, v) for u, v in zip(x, y)]))
                    o.setAttributes([a * a, a])
                    sk.addFeature(o, QgsFeatureSink.FastInsert)
            r["DISTORTION"] = did
        f = fields(
            *((k, QV.Double) for k in ("scale_factor", "area_factor", "lon", "lat"))
        )
        sk, mid = sink("MINIMUMS", f, T.Point)
        if sk is not None:
            if m.minimum_scale_factor is None:
                fb.pushInfo("No local minimums were found.")
            for v, (lo, la) in zip(
                *(
                    ()
                    if m.minimum_scale_factor is None
                    else (m.minimum_scale_factor, m.minimum_lonlat)
                )
            ):
                o = QgsFeature(f)
                o.setGeometry(
                    G.fromPointXY(
                        P(
                            *(
                                float(u[0])
                                for u in Z(
                                    sc, *m.w_to_m(*m.lonlat_to_w([lo], [la]))[:2]
                                )
                            )
                        )
                    )
                )
                o.setAttributes(
                    [float(v / rf), float((v / rf) ** 2), float(lo), float(la)]
                )
                sk.addFeature(o, QgsFeatureSink.FastInsert)
            r["MINIMUMS"] = mid
        for k, i in r.items():
            if c.willLoadLayerOnCompletion(i):
                c.layerToLoadOnCompletionDetails(i).name = (
                    f"{s.parameterDefinition(k).description()} ({m.name})"
                )
        return r
