import os
import sys

from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QMainWindow, QAction, QApplication, QMdiArea

from pivy.quarter import QuarterWidget

import freecad  # noqa: F401
import FreeCAD
import FreeCADGui
import Part

import bluemira.geometry._freecadapi as freecadapi


class MdiQuarterWidget(QuarterWidget):
    def __init__(self, parent, sharewidget):
        QuarterWidget.__init__(self, parent=parent, sharewidget=sharewidget)

    def minimumSizeHint(self):
        return QtCore.QSize(640, 480)


class MdiMainWindow(QMainWindow):
    def __init__(self, qApp):
        QMainWindow.__init__(self)
        self._firstwidget = None
        self._workspace = QMdiArea()
        self.setCentralWidget(self._workspace)
        self.setAcceptDrops(True)
        self.setWindowTitle("Pivy Quarter MDI example")

        filemenu = self.menuBar().addMenu("&File")
        windowmenu = self.menuBar().addMenu("&Windows")

        fileopenaction = QAction("&Create Box", self)
        createfaceaction = QAction("Create &Face", self)
        fileexitaction = QAction("E&xit", self)
        tileaction = QAction("Tile", self)
        cascadeaction = QAction("Cascade", self)

        filemenu.addAction(fileopenaction)
        filemenu.addAction(createfaceaction)
        filemenu.addAction(fileexitaction)
        windowmenu.addAction(tileaction)
        windowmenu.addAction(cascadeaction)

        self.connect(
            fileopenaction, QtCore.SIGNAL("triggered()"), self.createBoxInFreeCAD
        )
        self.connect(createfaceaction, QtCore.SIGNAL("triggered()"), self.createFace)
        self.connect(
            fileexitaction, QtCore.SIGNAL("triggered()"), QtGui.qApp.closeAllWindows
        )
        self.connect(
            tileaction, QtCore.SIGNAL("triggered()"), self._workspace.tileSubWindows
        )
        self.connect(
            cascadeaction,
            QtCore.SIGNAL("triggered()"),
            self._workspace.cascadeSubWindows,
        )

        windowmapper = QtCore.QSignalMapper(self)
        self.connect(
            windowmapper,
            QtCore.SIGNAL("mapped(QWidget *)"),
            self._workspace.setActiveSubWindow,
        )

        self.dirname = os.curdir

    def closeEvent(self, event):
        self._workspace.closeAllSubWindows()

    def createBoxInFreeCAD(self):
        d = FreeCAD.newDocument()
        o = d.addObject("Part::Box")
        d.recompute()
        s = FreeCADGui.subgraphFromObject(o)
        child = self.createMdiChild()
        child.show()
        child.setSceneGraph(s)

    def createFace(self):
        square_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        d = FreeCAD.newDocument()
        open_wire: Part.Wire = freecadapi.make_polygon(square_points)
        face = Part.Face([open_wire])
        o = d.addObject("Part::Feature")
        o.Shape = face
        d.recompute()
        s = FreeCADGui.subgraphFromObject(o)
        child = self.createMdiChild()
        child.show()
        child.setSceneGraph(s)

    def createMdiChild(self):
        widget = MdiQuarterWidget(None, self._firstwidget)
        self._workspace.addSubWindow(widget)
        if not self._firstwidget:
            self._firstwidget = widget
        return widget


def main():
    app = QApplication(sys.argv)
    FreeCADGui.setupWithoutGUI()
    mdi = MdiMainWindow(app)
    mdi.show()
    app.exec_()


if __name__ == "__main__":
    main()
