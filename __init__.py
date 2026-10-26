from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsApplication, QgsMapLayerType, QgsProcessingProvider


class Provider(QgsProcessingProvider):
    def id(s):
        return "geobica"

    def name(s):
        return "Geobica projection"

    def icon(s):
        from .algorithms import ICON_PATH

        return QIcon(ICON_PATH)

    def loadAlgorithms(s):
        from . import algorithms as a

        for c in (a.ProjectLayerAlgorithm, a.BoundaryGraticuleAlgorithm):
            s.addAlgorithm(c())


class Plugin:
    def __init__(s, i):
        s.iface = i
        s.provider = None
        s.actions = []

    def initProcessing(s):
        s.provider = Provider()
        QgsApplication.processingRegistry().addProvider(s.provider)

    def initGui(s):
        from .algorithms import ICON_PATH

        s.initProcessing()
        w = s.iface.mainWindow()
        for t, a, v, i in (
            ("Apply Optimized Conformal Projection…", "project_layer", 1, 1),
            ("Projection Boundary and Statistics…", "boundary_and_statistics", 0, 0),
        ):
            q = QAction(QIcon(ICON_PATH), t, w) if i else QAction(t, w)
            q.triggered.connect(lambda _=0, a=a, v=v: s._open(a, v))
            s.iface.addPluginToMenu("&Geobica Projection", q)
            s.actions.append(q)
            if i:
                s.iface.addToolBarIcon(q)

    def unload(s):
        for q in s.actions:
            s.iface.removePluginMenu("&Geobica Projection", q)
            s.iface.removeToolBarIcon(q)
        s.actions = []
        if s.provider:
            QgsApplication.processingRegistry().removeProvider(s.provider)
            s.provider = None

    def _open(s, a, v):
        import processing

        l = s.iface.activeLayer()
        processing.execAlgorithmDialog(
            "geobica:" + a,
            (
                {"INPUT": l.id()}
                if v and l is not None and l.type() == QgsMapLayerType.VectorLayer
                else {}
            ),
        )


def classFactory(i):
    return Plugin(i)
