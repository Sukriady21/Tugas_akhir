import sys
import os
import re
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QTableWidgetItem,QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel,QApplication,QMainWindow,QGraphicsScene,QTableWidgetItem
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
from RGB_testing import RGB
from RGB2HSV_testing import HSV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Menggunakan File UI yang merupakan file desain aplikasi
        loadUi(r"D:\skripsi jilid 2\sukriadi\Code\form_utama.ui", self)

        # Inisialisasi variabel untuk alagoritma kNN
        self.knn_Naga = None

        # Inisialisasi variabel untuk menyimpan nilai K awal
        self.rgb_object = RGB(k=0)
        self.hsv_object = HSV(k=0)

        # inisialisasi objek menyimpan hasil prediksi
        self.predictions_RGB = []
        self.predictions_HSV = []

        self.prediksi_salah = 0
        self.prediksi_benar = 0
        self.label_asli_data_uji = []  # Inisialisasi dengan data aktual
        self.predictions_HSV = []

        # inisialisasi variabel untuk menyimpan label kematangan dari ekstraksi warna
        self.label_RGB = QLabel(self)
        self.label_HSV = QLabel(self)

        # Menghubungkan tombol-tombol dengan func def
        self.btnBacaDataLatih.clicked.connect(self.bacaDataLatih)
        self.btnProsesPelatihan.clicked.connect(self.prosesPelatihan)
        self.btnBacaDataUji.clicked.connect(self.bacaDataUji)
        self.btnProsesEkstraksi.clicked.connect(self.prosesEkstraksiDataUji)
        self.label_RGB = QLabel(self)
        self.label_HSV = QLabel(self)
        self.btnUji_KNN_RGB.clicked.connect(self.uji_KNN_RGB)
        self.btnUji_KNN_HSV.clicked.connect(self.uji_KNN_HSV)
        self.editPrediksi_benar.setText("0")
        self.editPrediksi_salah.setText("0")
        self.editAkurasiKNN.setText("0")
        self.btnReset.clicked.connect(self.resetTabelHasilPengujian)
        self.nilai_K.textChanged.connect(self.updateNilaiK)
        self.btnMatrixRGB.clicked.connect(self.hitung_confusion_matrix_KNN_RGB)
        self.btnF1scoreRGB.clicked.connect(self.hitung_F1score_KNN_RGB)
        self.btnMatrixHSV.clicked.connect(self.hitung_confusion_matrix_KNN_HSV)
        self.btnF1scoreHSV.clicked.connect(self.hitung_F1score_KNN_HSV)
        self.btnResetMatrix.clicked.connect(self.reset_confusion_matrix_and_F1score)
        self.btnSalinKeDataLatih.clicked.connect(self.salinKeExcelDataLatih)
        self.btnSalinKeDataUji.clicked.connect(self.salinKeExcelDataUji)
        self.btnSalinKePengujian.clicked.connect(self.salinKeExcelDataPrediksi)

        # menyimpan label asli dari data uji
        self.label_asli_data_uji = [
            "matang","matang","matang","matang","matang","matang","matang","matang","matang","setengah_matang",
            "setengah_matang","setengah_matang","setengah_matang","setengah_matang","setengah_matang",
            "belum_matang","belum_matang","belum_matang","belum_matang","belum_matang","belum_matang",
            "belum_matang","setengah_matang","setengah_matang","setengah_matang","setengah_matang",
            "matang","belum_matang","belum_matang","belum_matang",
        ]

    # Fungsi membaca data latih
    def bacaDataLatih(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Pilih Folder Data Latih")
        if folder_path:
            self.knn_Naga = RGB(k=0)
            self.knn_Naga.data_latih_dir = folder_path
            self.btnProsesPelatihan.setEnabled(True)
        else:
            QMessageBox.warning(self, "Peringatan", "Gagal membaca folder data latih.")

    # Fungsi Proses Pelatihan Data
    def prosesPelatihan(self):
        if not self.knn_Naga:
            QMessageBox.warning(self, "Peringatan", "Folder data latih belum dipilih.")
            return

        self.knn_Naga.image_latih_paths = self.knn_Naga.load_image_paths(self.knn_Naga.data_latih_dir)
        if not self.knn_Naga.image_latih_paths:
            QMessageBox.warning(self,"Peringatan","Tidak ada gambar yang ditemukan dalam folder data latih.",)
            return

        self.knn_Naga.image_latih_paths.sort(key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))
        self.knn_Naga.data_latih_labels = self.knn_Naga.get_data_labels(self.knn_Naga.image_latih_paths)
        self.knn_Naga.data_latih_features = self.knn_Naga.extract_features(self.knn_Naga.image_latih_paths)

        # Update GUI
        self.editJumlahDataLatih.setText(str(len(self.knn_Naga.image_latih_paths)))
        self.editJumlahDataLatih_b.setText(str(self.knn_Naga.data_latih_labels.count("belum_matang")))
        self.editJumlahDataLatih_s.setText(str(self.knn_Naga.data_latih_labels.count("setengah_matang")))
        self.editJumlahDataLatih_m.setText(str(self.knn_Naga.data_latih_labels.count("matang")))

        # Display data latih dan nilai ekstraksi warna
        self.tblDataLatih.setRowCount(len(self.knn_Naga.image_latih_paths))
        self.tblDataLatih.setColumnCount(8)
        self.tblDataLatih.setHorizontalHeaderLabels(["Nama File", "R", "G", "B", "H", "S", "V", "Kelas"])

        # Mengelompokkan paths berdasarkan kelas
        paths_by_class = {}
        for path, label in zip(
            self.knn_Naga.image_latih_paths, self.knn_Naga.data_latih_labels
        ):
            if label not in paths_by_class:
                paths_by_class[label] = []
            paths_by_class[label].append(path)

        # Mengisi tabel dengan nama file, kelas, dan rata-rata warna
        current_row = 0
        for label, paths in paths_by_class.items():
            # Mengurutkan berdasarkan nomor urut di nama file
            paths.sort(key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))
            for path in paths:
                r, g, b = self.knn_Naga.get_rgb(path)
                # Menggunakan metode dari kelas HSV
                h, s, v = self.hsv_object.rgb_to_hsv(r.mean(), g.mean(), b.mean())
                file_name = os.path.basename(path)
                self.tblDataLatih.setItem(current_row, 0, QTableWidgetItem(file_name))
                self.tblDataLatih.setItem(current_row, 1, QTableWidgetItem(f"{r.mean():.2f}"))
                self.tblDataLatih.setItem(current_row, 2, QTableWidgetItem(f"{g.mean():.2f}"))
                self.tblDataLatih.setItem(current_row, 3, QTableWidgetItem(f"{b.mean():.2f}"))
                self.tblDataLatih.setItem(current_row, 4, QTableWidgetItem(f"{h:.2f}"))
                self.tblDataLatih.setItem(current_row, 5, QTableWidgetItem(f"{s:.2f}"))
                self.tblDataLatih.setItem(current_row, 6, QTableWidgetItem(f"{v:.2f}"))
                self.tblDataLatih.setItem(current_row, 7, QTableWidgetItem(label))
                current_row += 1

    def salinKeExcelDataLatih(self):
        clipboard = QtWidgets.QApplication.clipboard()
        selected_rows = self.tblDataLatih.selectionModel().selectedRows()
        data_to_copy = []

        if selected_rows:
            # Copy selected rows
            for row in selected_rows:
                row_data = [
                    self.tblDataLatih.item(row.row(), col).text()
                    for col in range(self.tblDataLatih.columnCount())
                ]
                data_to_copy.append("\t".join(row_data))
        else:
            # Copy all rows
            for row in range(self.tblDataLatih.rowCount()):
                row_data = [
                    self.tblDataLatih.item(row, col).text()
                    for col in range(self.tblDataLatih.columnCount())
                ]
                data_to_copy.append("\t".join(row_data))

        data = "\n".join(data_to_copy)
        clipboard.setText(data)

    # Fungsi membaca data uji
    def bacaDataUji(self):
        # Dapatkan path folder data uji
        folder_path = QFileDialog.getExistingDirectory(self, "Pilih Folder Data Uji")
        if folder_path:
            self.knn_Naga.data_uji_dir = folder_path
            self.btnUji_KNN_RGB.setEnabled(True)
            self.btnUji_KNN_HSV.setEnabled(True)
        else:
            QMessageBox.warning(self, "Peringatan", "Gagal membaca folder data uji.")

    # Fungsi proses ekstrasi warna data uji
    def prosesEkstraksiDataUji(self):
        if not self.knn_Naga:
            QMessageBox.warning(self, "Peringatan", "Model KNN belum dilatih.")
            return

        if not self.knn_Naga.data_uji_dir:
            QMessageBox.warning(self, "Peringatan", "Folder data uji belum dipilih.")
            return

        image_uji_paths = self.knn_Naga.load_image_paths(self.knn_Naga.data_uji_dir)
        if not image_uji_paths:
            QMessageBox.warning(self,"Peringatan","Tidak ada gambar yang ditemukan dalam folder data uji.",)
            return

        self.tblDataUji.setRowCount(len(image_uji_paths))
        self.tblDataUji.setColumnCount(7)
        self.tblDataUji.setHorizontalHeaderLabels(["Nama File", "R", "G", "B", "H", "S", "V"])

        # Urutkan image paths berdasarkan nama file
        image_uji_paths.sort(key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))

        # Set jumlah data uji ke widget editJumlahDataUji
        self.editJumlahDataUji.setText(str(len(image_uji_paths)))

        for i, path in enumerate(image_uji_paths):
            r, g, b = self.knn_Naga.get_rgb(path)
            h, s, v = self.hsv_object.rgb_to_hsv(r.mean(), g.mean(), b.mean())
            file_name = os.path.basename(path)
            self.tblDataUji.setItem(i, 0, QTableWidgetItem(file_name))
            self.tblDataUji.setItem(i, 1, QTableWidgetItem(f"{r.mean():.2f}"))
            self.tblDataUji.setItem(i, 2, QTableWidgetItem(f"{g.mean():.2f}"))
            self.tblDataUji.setItem(i, 3, QTableWidgetItem(f"{b.mean():.2f}"))
            self.tblDataUji.setItem(i, 4, QTableWidgetItem(f"{h:.2f}"))
            self.tblDataUji.setItem(i, 5, QTableWidgetItem(f"{s:.2f}"))
            self.tblDataUji.setItem(i, 6, QTableWidgetItem(f"{v:.2f}"))

    def salinKeExcelDataUji(self):
        clipboard = QtWidgets.QApplication.clipboard()
        selected_rows = self.tblDataUji.selectionModel().selectedRows()
        data_to_copy = []

        if selected_rows:
            # Copy selected rows
            for row in selected_rows:
                row_data = [
                    self.tblDataUji.item(row.row(), col).text()
                    for col in range(self.tblDataUji.columnCount())
                ]
                data_to_copy.append("\t".join(row_data))
        else:
            # Copy all rows
            for row in range(self.tblDataUji.rowCount()):
                row_data = [
                    self.tblDataUji.item(row, col).text()
                    for col in range(self.tblDataUji.columnCount())
                ]
                data_to_copy.append("\t".join(row_data))

        data = "\n".join(data_to_copy)
        clipboard.setText(data)

    # Fungsi untuk mengizinkan sistem mengubah nilai K secara bebas
    def updateNilaiK(self, nilai):
        if self.knn_Naga:
            self.knn_Naga.k = nilai
            self.hsv_object.k = nilai

    # fungsi pengujian KNN Ekstrasi Warna-RGB
    def uji_KNN_RGB(self):
        if not self.knn_Naga:
            QMessageBox.warning(self, "Peringatan", "Model KNN belum dilatih.")
            return

        if not self.knn_Naga.data_uji_dir:
            QMessageBox.warning(self, "Peringatan", "Folder data uji belum dipilih.")
            return

        image_uji_paths = self.knn_Naga.load_image_paths(self.knn_Naga.data_uji_dir)
        if not image_uji_paths:
            QMessageBox.warning(self,"Peringatan","Tidak ada gambar yang ditemukan dalam folder data uji.",)
            return

        total_data_uji = len(image_uji_paths)
        prediksi_benar = 0

        # Mengurutkan nama file
        image_uji_paths.sort(key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))

        for index, (path, kelas_asli) in enumerate(
            zip(image_uji_paths, self.label_asli_data_uji), start=0
        ):
            prediksi = self.knn_Naga.predict(path)
            self.predictions_RGB.append(prediksi)

            # Menampilkan hasil ke tabel
            file_name = os.path.basename(path)
            row_position = self.tblHasilPengujian.rowCount()
            self.tblHasilPengujian.insertRow(row_position)
            self.tblHasilPengujian.setItem(row_position, 0, QTableWidgetItem(file_name))
            self.tblHasilPengujian.setItem(row_position, 1, QTableWidgetItem(self.label_asli_data_uji[index - 0]))
            self.tblHasilPengujian.setItem(row_position, 2, QTableWidgetItem(prediksi))

            if prediksi == kelas_asli:
                prediksi_benar += 1

        prediksi_salah = total_data_uji - prediksi_benar
        # Menghitung akurasi
        akurasi = ((total_data_uji - prediksi_salah) / total_data_uji) * 100

        self.editPrediksi_benar.setText(str(prediksi_benar))
        self.editPrediksi_salah.setText(str(prediksi_salah))
        self.editAkurasiKNN.setText(f"{akurasi:.2f}%")

    # fungsi pengujian KNN Ekstrasi Warna-HSV
    def uji_KNN_HSV(self):
        if not self.knn_Naga:
            QMessageBox.warning(self, "Peringatan", "Model KNN belum dilatih.")
            return

        if not self.knn_Naga.data_uji_dir:
            QMessageBox.warning(self, "Peringatan", "Folder data uji belum dipilih.")
            return

        if not self.hsv_object:
            QMessageBox.warning(self, "Peringatan", "Objek HSV belum diinisialisasi.")
            return

        image_uji_paths = self.knn_Naga.load_image_paths(self.knn_Naga.data_uji_dir)
        if not image_uji_paths:
            QMessageBox.warning(self,"Peringatan","Tidak ada gambar yang ditemukan dalam folder data uji.",)
            return

        total_data_uji = len(image_uji_paths)
        self.prediksi_benar = 0

        # Mengurutkan nama file
        image_uji_paths.sort(key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))

        for index, (path, kelas_asli) in enumerate(
            zip(image_uji_paths, self.label_asli_data_uji), start=0
        ):
            # Proses Konversi RGB ke HSV
            prediksi = self.hsv_object.predict(path)
            self.predictions_HSV.append(prediksi)

            # Menampilkan hasil ke tabel
            file_name = os.path.basename(path)
            row_position = self.tblHasilPengujian.rowCount()
            self.tblHasilPengujian.insertRow(row_position)
            self.tblHasilPengujian.setItem(row_position, 0, QTableWidgetItem(file_name))
            self.tblHasilPengujian.setItem(row_position, 1, QTableWidgetItem(self.label_asli_data_uji[index - 0]))
            self.tblHasilPengujian.setItem(row_position, 2, QTableWidgetItem(prediksi))

            if prediksi == kelas_asli:
                self.prediksi_benar += 1

        self.prediksi_salah = total_data_uji - self.prediksi_benar
        # Menghitung akurasi
        akurasi = ((total_data_uji - self.prediksi_salah) / total_data_uji) * 100

        self.editPrediksi_benar.setText(str(self.prediksi_benar))
        self.editPrediksi_salah.setText(str(self.prediksi_salah))
        self.editAkurasiKNN.setText(f"{akurasi:.2f}%")

    def salinKeExcelDataPrediksi(self):
        clipboard = QtWidgets.QApplication.clipboard()
        selected_rows = self.tblHasilPengujian.selectionModel().selectedRows()
        data_to_copy = []

        if selected_rows:
            # Copy selected rows
            for row in selected_rows:
                row_data = [
                    self.tblHasilPengujian.item(row.row(), col).text()
                    for col in range(self.tblHasilPengujian.columnCount())
                ]
                data_to_copy.append("\t".join(row_data))
        else:
            # Copy all rows
            for row in range(self.tblHasilPengujian.rowCount()):
                row_data = [
                    self.tblHasilPengujian.item(row, col).text()
                    for col in range(self.tblHasilPengujian.columnCount())
                ]
                data_to_copy.append("\t".join(row_data))

        data = "\n".join(data_to_copy)
        clipboard.setText(data)

    # Fungsi reset tabel hasil prediksi
    def resetTabelHasilPengujian(self):
        self.tblHasilPengujian.clearContents()
        self.tblHasilPengujian.setRowCount(0)

    # Menghitung confusion matrix untuk model KNN-RGB
    def hitung_confusion_matrix_KNN_RGB(self, table):

        conf_matrix = confusion_matrix(self.label_asli_data_uji,self.predictions_RGB,labels=np.unique(self.label_asli_data_uji),)

        # Ensure the table widgets have the necessary rows and columns
        self.tblMatrix_B.setRowCount(2)
        self.tblMatrix_B.setColumnCount(3)

        self.tblMatrix_S.setRowCount(2)
        self.tblMatrix_S.setColumnCount(3)

        self.tblMatrix_M.setRowCount(2)
        self.tblMatrix_M.setColumnCount(3)

        # Update the table widgets with the confusion matrix values
        classes = ["TP", "FP", "FN"]

        for i, label in enumerate(np.unique(self.label_asli_data_uji)):
            # Select the appropriate table widget based on the class label
            if label == "belum_matang":
                table_widget = self.tblMatrix_B
            elif label == "setengah_matang":
                table_widget = self.tblMatrix_S
            elif label == "matang":
                table_widget = self.tblMatrix_M
            else:
                continue  # Skip if label is not recognized

            # Update the first row of the table widget with class names
            for j, cls in enumerate(classes):
                item = QTableWidgetItem(cls)
                table_widget.setItem(0, j, item)

            # Update the second row of the table widget with TP, FP, FN values
            TP = conf_matrix[i, i]
            FP = np.sum(conf_matrix[:, i]) - TP
            FN = np.sum(conf_matrix[i, :]) - TP

            values = [TP, FP, FN]
            for j, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                table_widget.setItem(1, j, item)

    def hitung_F1score_KNN_RGB(self):
        # Menghitung F1-Score untuk model KNN-RGB
        f1_per_kelas = []

        # Inisialisasi confusion matrix
        cm = np.zeros((3, 3), dtype=int)

        # Pengaturan elemen matriks kebingungan berdasarkan label yang diurutkan
        for i in range(len(self.label_asli_data_uji)):
            true_label = self.label_asli_data_uji[i]
            pred_label = self.predictions_RGB[i]
            if true_label == "belum_matang" and pred_label == "belum_matang":
                cm[0, 0] += 1
            elif true_label == "setengah_matang" and pred_label == "belum_matang":
                cm[0, 1] += 1
            elif true_label == "belum_matang" and pred_label == "matang":
                cm[0, 2] += 1
            elif true_label == "belum_matang" and pred_label == "setengah_matang":
                cm[1, 0] += 1
            elif true_label == "setengah_matang" and pred_label == "setengah_matang":
                cm[1, 1] += 1
            elif true_label == "setengah_matang" and pred_label == "matang":
                cm[1, 2] += 1
            elif true_label == "matang" and pred_label == "belum_matang":
                cm[2, 0] += 1
            elif true_label == "matang" and pred_label == "setengah_matang":
                cm[2, 1] += 1
            elif true_label == "matang" and pred_label == "matang":
                cm[2, 2] += 1

        # Iterate over classes and calculate F1-score
        labels = np.unique(self.label_asli_data_uji)
        for i, label in enumerate(labels):
            # Hitung TP, FP, FN untuk setiap kelas
            TP = np.diag(cm)
            FP = cm.sum(axis=0) - TP
            FN = cm.sum(axis=1) - TP

            # Hitung akurasi
            jumlah_data_benar = np.sum(TP)
            jumlah_data_uji = len(self.label_asli_data_uji)
            accuracy = (jumlah_data_benar / jumlah_data_uji) * 100

            # Hitung presisi untuk setiap kelas
            precision_per_kelas = TP / (FP + TP) * 100
            precision_per_kelas = np.nan_to_num(precision_per_kelas)

            # Hitung recall untuk setiap kelas
            recall_per_kelas = TP / (FN + TP) * 100
            recall_per_kelas = np.nan_to_num(recall_per_kelas)

            # Hitung F1-Score untuk setiap kelas
            f1_per_kelas = (2 * (precision_per_kelas * recall_per_kelas) / (precision_per_kelas + recall_per_kelas))
            f1_per_kelas = np.nan_to_num(f1_per_kelas)

            # Tampilkan hasil untuk setiap kelas
            for i, label in enumerate(np.unique(self.label_asli_data_uji)):
                precision_kelas = precision_per_kelas[i]
                recall_kelas = recall_per_kelas[i]
                f1_kelas = f1_per_kelas[i]

                if label == "belum_matang":
                    self.editJumlahPresisi_B.setText("{:.2f}%".format(recall_kelas))
                    self.editJumlahRecall_B.setText("{:.2f}%".format(precision_kelas))
                    self.editF1score_B.setText("{:.2f}%".format(f1_kelas))
                elif label == "matang":
                    self.editJumlahPresisi_S.setText("{:.2f}%".format(recall_kelas))
                    self.editJumlahRecall_S.setText("{:.2f}%".format(precision_kelas))
                    self.editJumlahF1score_s.setText("{:.2f}%".format(f1_kelas))
                elif label == "setengah_matang":
                    self.editJumlahPresisi_M.setText("{:.2f}%".format(recall_kelas))
                    self.editJumlahRecall_M.setText("{:.2f}%".format(precision_kelas))
                    self.editJumlahF1score_M.setText("{:.2f}%".format(f1_kelas))

            # Calculate average F1-score and update the widget
            self.editJumlahAkurasi.setText("{:.2f}%".format(accuracy))
            f1 = np.mean(f1_per_kelas)
            self.editRata2F1score.setText("{:.2f}%".format(f1))

        # Plot heatmap
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", xticklabels=labels, yticklabels=labels, annot_kws={"size": 20})
        plt.xlabel("Predicted Label", fontsize=14)
        plt.ylabel("True Label", fontsize=14)
        plt.title("Confusion Matrix", fontsize=16)

        # Konversi gambar Matplotlib menjadi QPixmap
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        width, height = canvas.get_width_height()
        img = np.frombuffer(canvas.buffer_rgba(), dtype="uint8").reshape(height, width, 4)  # Ubah ke tostring_rgba() dan sesuaikan dimensi array
        qt_img = QPixmap.fromImage(QImage(img, width, height, QImage.Format_RGBA8888))  # Format_RGBA8888

        # Tampilkan QPixmap di QGraphicsView
        scene = QGraphicsScene()
        scene.addPixmap(qt_img)
        self.graphicsView.setScene(scene)
        self.graphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        # Membersihkan plot Matplotlib untuk menghindari tumpang tindih pada plot berikutnya
        plt.clf()

    def hitung_confusion_matrix_KNN_HSV(self, table):
        # Menghitung confusion matrix untuk model KNN-RGB
        conf_matrix = confusion_matrix(self.label_asli_data_uji,self.predictions_HSV,labels=np.unique(self.label_asli_data_uji),)

        # Ensure the table widgets have the necessary rows and columns
        self.tblMatrix_B.setRowCount(2)
        self.tblMatrix_B.setColumnCount(3)

        self.tblMatrix_S.setRowCount(2)
        self.tblMatrix_S.setColumnCount(3)

        self.tblMatrix_M.setRowCount(2)
        self.tblMatrix_M.setColumnCount(3)

        # Update the table widgets with the confusion matrix values
        classes = ["TP", "FP", "FN"]

        for i, label in enumerate(np.unique(self.label_asli_data_uji)):
            # Select the appropriate table widget based on the class label
            if label == "belum_matang":
                table_widget = self.tblMatrix_B
            elif label == "setengah_matang":
                table_widget = self.tblMatrix_S
            elif label == "matang":
                table_widget = self.tblMatrix_M
            else:
                continue  # Skip if label is not recognized

            # Update the first row of the table widget with class names
            for j, cls in enumerate(classes):
                item = QTableWidgetItem(cls)
                table_widget.setItem(0, j, item)

            # Update the second row of the table widget with TP, FP, FN values
            TP = conf_matrix[i, i]
            FP = np.sum(conf_matrix[:, i]) - TP
            FN = np.sum(conf_matrix[i, :]) - TP

            values = [TP, FP, FN]
            for j, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                table_widget.setItem(1, j, item)

    def hitung_F1score_KNN_HSV(self):
        # Menghitung F1-Score untuk model KNN-RGB
        f1_per_kelas = []

        # Inisialisasi confusion matrix
        cm = np.zeros((3, 3), dtype=int)

        # Pengaturan elemen matriks kebingungan berdasarkan label yang diurutkan
        for i in range(len(self.label_asli_data_uji)):
            true_label = self.label_asli_data_uji[i]
            pred_label = self.predictions_HSV[i]
            if true_label == "belum_matang" and pred_label == "belum_matang":
                cm[0, 0] += 1
            elif true_label == "setengah_matang" and pred_label == "belum_matang":
                cm[0, 1] += 1
            elif true_label == "belum_matang" and pred_label == "matang":
                cm[0, 2] += 1
            elif true_label == "belum_matang" and pred_label == "setengah_matang":
                cm[1, 0] += 1
            elif true_label == "setengah_matang" and pred_label == "setengah_matang":
                cm[1, 1] += 1
            elif true_label == "setengah_matang" and pred_label == "matang":
                cm[1, 2] += 1
            elif true_label == "matang" and pred_label == "belum_matang":
                cm[2, 0] += 1
            elif true_label == "matang" and pred_label == "setengah_matang":
                cm[2, 1] += 1
            elif true_label == "matang" and pred_label == "matang":
                cm[2, 2] += 1

        # Iterate over classes and calculate F1-score
        labels = np.unique(self.label_asli_data_uji)
        for i, label in enumerate(labels):
            # Hitung TP, FP, FN untuk setiap kelas
            TP = np.diag(cm)
            FP = cm.sum(axis=0) - TP
            FN = cm.sum(axis=1) - TP

            # Hitung akurasi
            jumlah_data_benar = np.sum(TP)
            jumlah_data_uji = len(self.label_asli_data_uji)
            accuracy = (jumlah_data_benar / jumlah_data_uji) * 100

            # Hitung presisi untuk setiap kelas
            precision_per_kelas = TP / (FP + TP) * 100
            precision_per_kelas = np.nan_to_num(precision_per_kelas)

            # Hitung recall untuk setiap kelas
            recall_per_kelas = TP / (FN + TP) * 100
            recall_per_kelas = np.nan_to_num(recall_per_kelas)

            # Hitung F1-Score untuk setiap kelas
            f1_per_kelas = (2 * (precision_per_kelas * recall_per_kelas) / (precision_per_kelas + recall_per_kelas))
            f1_per_kelas = np.nan_to_num(f1_per_kelas)

            # Tampilkan hasil untuk setiap kelas
            for i, label in enumerate(np.unique(self.label_asli_data_uji)):
                precision_kelas = precision_per_kelas[i]
                recall_kelas = recall_per_kelas[i]
                f1_kelas = f1_per_kelas[i]

                if label == "belum_matang":
                    self.editJumlahPresisi_B.setText("{:.2f}%".format(recall_kelas))
                    self.editJumlahRecall_B.setText("{:.2f}%".format(precision_kelas))
                    self.editF1score_B.setText("{:.2f}%".format(f1_kelas))
                elif label == "matang":
                    self.editJumlahPresisi_S.setText("{:.2f}%".format(recall_kelas))
                    self.editJumlahRecall_S.setText("{:.2f}%".format(precision_kelas))
                    self.editJumlahF1score_s.setText("{:.2f}%".format(f1_kelas))
                elif label == "setengah_matang":
                    self.editJumlahPresisi_M.setText("{:.2f}%".format(recall_kelas))
                    self.editJumlahRecall_M.setText("{:.2f}%".format(precision_kelas))
                    self.editJumlahF1score_M.setText("{:.2f}%".format(f1_kelas))

            # Calculate average F1-score and update the widget
            self.editJumlahAkurasi.setText("{:.2f}%".format(accuracy))
            f1 = np.mean(f1_per_kelas)
            self.editRata2F1score.setText("{:.2f}%".format(f1))

        # Plot heatmap
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", xticklabels=labels, yticklabels=labels, annot_kws={"size": 20})
        plt.xlabel("Predicted Label", fontsize=14)
        plt.ylabel("True Label", fontsize=14)
        plt.title("Confusion Matrix", fontsize=16) 

        # Konversi gambar Matplotlib menjadi QPixmap
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        width, height = canvas.get_width_height()
        img = np.frombuffer(canvas.buffer_rgba(), dtype="uint8").reshape(height, width, 4)  # Ubah ke tostring_rgba() dan sesuaikan dimensi array
        qt_img = QPixmap.fromImage(QImage(img, width, height, QImage.Format_RGBA8888))  # Format_RGBA8888

        # Tampilkan QPixmap di QGraphicsView
        scene = QGraphicsScene()
        scene.addPixmap(qt_img)
        self.graphicsView.setScene(scene)
        self.graphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        # Membersihkan plot Matplotlib untuk menghindari tumpang tindih pada plot berikutnya
        plt.clf()

    def reset_confusion_matrix_and_F1score(self):
        # Reset confusion matrix values
        self.tblMatrix_B.clearContents()
        self.tblMatrix_S.clearContents()
        self.tblMatrix_M.clearContents()

        # Reset F1-score and related values
        self.editJumlahAkurasi.setText("")
        self.editJumlahPresisi_B.setText("")
        self.editJumlahRecall_B.setText("")
        self.editF1score_B.setText("")

        self.editJumlahPresisi_S.setText("")
        self.editJumlahRecall_S.setText("")
        self.editJumlahF1score_s.setText("")

        self.editJumlahPresisi_M.setText("")
        self.editJumlahRecall_M.setText("")
        self.editJumlahF1score_M.setText("")

        # Reset confusion matrix plot
        self.graphicsView.setScene(None)

        # Reset predictions
        self.predictions_RGB = []
        self.predictions_HSV = []
        self.predictions_Lab = []

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
