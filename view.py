import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QTableView, QGraphicsView, QGraphicsScene, QMenu, QAction
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QPixmap, QImage

from simulate import create_star_image


class StarImageViewer(QMainWindow):
    def __init__(self):
        super(StarImageViewer, self).__init__()

        # central widget
        central_widget = QWidget(self)
        main_layout = QHBoxLayout(central_widget)

        # image
        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setMinimumWidth(512)
        main_layout.addWidget(self.view)

        # right layout for input and button
        right_layout = QVBoxLayout()

        # table
        self.table = QTableView(self)
        right_layout.addWidget(self.table)

        # input
        settings = ["ra", "de", "roll", "h", "w", 'fov', "sigma_g", "prob_p"]
        self.inputs = {}
        input_layout = QGridLayout()
        for i, name in enumerate(settings):
            row, col = i // 3, (i % 3) * 2
            input_layout.addWidget(QLabel(name), row, col)
            self.inputs[name] = QLineEdit(self)
            input_layout.addWidget(self.inputs[name], row, col+1)

        right_layout.addLayout(input_layout)
        
        # button
        btn_layout = QHBoxLayout()
        btn1 = QPushButton('draw', self)
        btn1.clicked.connect(self.draw_img)
        btn_layout.addWidget(btn1)

        btn2 = QPushButton('save', self)
        btn2.clicked.connect(self.save_img)
        btn_layout.addWidget(btn2)
        
        right_layout.addLayout(btn_layout)
        main_layout.addLayout(right_layout)
        
        # menu bar  
        menubar = self.menuBar()
        file_menu = QMenu('File', self)
        view_menu = QMenu('View', self)
        help_menu = QMenu('Help', self)
        menubar.addMenu(file_menu)
        menubar.addMenu(view_menu)
        menubar.addMenu(help_menu)

        # set central widget
        self.setCentralWidget(central_widget)
        self.setWindowTitle('star image simulator')
        self.setGeometry(300, 300, 800, 512)
        self.show()

    def draw_img(self):
        # get input
        ra, de, roll = float(self.inputs['ra'].text()), float(self.inputs['de'].text()), float(self.inputs['roll'].text())
        # h, w = int(self.inputs['h'].text()), int(self.inputs['w'].text())
        # fov = float(self.inputs['fov'].text())
        sigma_g = float(self.inputs['sigma_g'].text())
        prob_p = float(self.inputs['prob_p'].text())

        print(ra, de, roll, sigma_g, prob_p)

        # simulate star img
        img, df = create_star_image(ra, de, roll, sigma_g, prob_p, simulate_test=True)
        h, w = img.shape

        # show img
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self.scene.clear()
        self.scene.addPixmap(pixmap)

        # show table
        model = PandasModel(df)
        self.table.setModel(model)


    def save_img(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "JPEG Files (*.jpg);;PNG Files (*.png)")
        if file_path:
            pixmap = self.image_label.pixmap()
            if pixmap:
                pixmap.save(file_path)
                print("图片已保存到：", file_path)


class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = StarImageViewer()
    sys.exit(app.exec_())