# -*- coding: utf-8 -*-
#
# DotDotGoose
#
# --------------------------------------------------------------------------
#
# This file is part of the DotDotGoose application.
# DotDotGoose was forked from the Neural Network Image Classifier (Nenetic).
#
# DotDotGoose is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DotDotGoose is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with with this software.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------
from PyQt6 import QtCore, QtGui, QtWidgets


class DetectionWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object)  # Params (DetectionResult)

    def __init__(self, detector, image_array, region, sensitivity, polarity, existing_points, dedup_radius):
        QtCore.QThread.__init__(self)
        self.detector = detector
        self.image_array = image_array
        self.region = region
        self.sensitivity = sensitivity
        self.polarity = polarity
        self.existing_points = existing_points
        self.dedup_radius = dedup_radius

    def run(self):
        result = self.detector.detect(self.image_array, region=self.region, sensitivity=self.sensitivity,
                                       polarity=self.polarity, existing_points=self.existing_points,
                                       dedup_radius=self.dedup_radius)
        self.result_ready.emit(result)


class DetectDialog(QtWidgets.QDialog):

    def __init__(self, canvas, has_selection, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle(self.tr('Auto-Detect Points'))
        self.setModal(True)
        self.canvas = canvas
        self.worker = None

        self.sliderSensitivity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sliderSensitivity.setRange(0, 100)
        self.sliderSensitivity.setValue(50)
        self.labelSensitivityValue = QtWidgets.QLabel('50')
        self.sliderSensitivity.valueChanged.connect(lambda value: self.labelSensitivityValue.setText(str(value)))

        self.radioButtonBright = QtWidgets.QRadioButton(self.tr('Bright objects on dark background'))
        self.radioButtonDark = QtWidgets.QRadioButton(self.tr('Dark objects on bright background'))
        self.radioButtonBright.setChecked(True)

        self.checkBoxRestrict = QtWidgets.QCheckBox(self.tr('Restrict to selected region'))
        self.checkBoxRestrict.setChecked(has_selection)
        self.checkBoxRestrict.setEnabled(has_selection)

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setRange(0, 0)
        self.progressBar.hide()

        self.pushButtonRun = QtWidgets.QPushButton(self.tr('Run'))
        self.pushButtonCancel = QtWidgets.QPushButton(self.tr('Cancel'))
        self.pushButtonRun.clicked.connect(self.run_detection)
        self.pushButtonCancel.clicked.connect(self.cancel)
        self.pushButtonRun.setIcon(QtGui.QIcon('icons:detect.svg'))
        self.pushButtonCancel.setIcon(QtGui.QIcon('icons:cancel.svg'))

        sensitivity_row = QtWidgets.QHBoxLayout()
        sensitivity_row.addWidget(QtWidgets.QLabel(self.tr('Sensitivity')))
        sensitivity_row.addWidget(self.sliderSensitivity)
        sensitivity_row.addWidget(self.labelSensitivityValue)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.pushButtonRun)
        button_row.addWidget(self.pushButtonCancel)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(sensitivity_row)
        layout.addWidget(self.radioButtonBright)
        layout.addWidget(self.radioButtonDark)
        layout.addWidget(self.checkBoxRestrict)
        layout.addWidget(self.progressBar)
        layout.addLayout(button_row)
        self.setLayout(layout)
        self.resize(320, self.sizeHint().height())

    def cancel(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.set_running(False)
        else:
            self.close()

    def run_detection(self):
        canvas = self.canvas
        if canvas.image_cache['data'] is None or canvas.current_image_name is None or canvas.current_class_name is None:
            return
        region = None
        if self.checkBoxRestrict.isChecked():
            region = canvas.last_region
        polarity = 'dark' if self.radioButtonDark.isChecked() else 'bright'
        existing = canvas.points[canvas.current_image_name].get(canvas.current_class_name, [])

        self.set_running(True)
        self.worker = DetectionWorker(canvas.detector, canvas.image_cache['data'], region,
                                       self.sliderSensitivity.value(), polarity, existing,
                                       canvas.ui['point']['radius'])
        self.worker.result_ready.connect(self.on_result)
        self.worker.start()

    def on_result(self, result):
        self.canvas._commit_detected_points(result.points)
        self.set_running(False)
        self.close()

    def set_running(self, running):
        self.progressBar.setVisible(running)
        self.pushButtonRun.setEnabled(not running)
        self.pushButtonCancel.setText(self.tr('Cancel') if running else self.tr('Close'))
