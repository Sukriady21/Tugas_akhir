import os
import cv2
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns

class HSV:
    def __init__(self, k):
        self.k = k
        self.data_latih_dir = r"D:\skripsi jilid 2\sukriadi\data_latih"
        self.image_latih_paths = self.load_image_paths(self.data_latih_dir)
        self.image_latih_paths.sort(key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))
        self.data_latih_labels = self.get_data_labels(self.image_latih_paths)
        self.data_latih_features = self.extract_features(self.image_latih_paths)

    def load_image_paths(self, directory):
        image_paths = []
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".png"):
                        image_paths.append(os.path.join(folder_path, file_name))
        return image_paths

    def get_data_labels(self, image_paths):
        labels = []
        for path in image_paths:
            label = os.path.basename(os.path.dirname(path))
            labels.append(label)
        return labels

    @staticmethod
    def rgb_to_hsv(r, g, b):
        # Normalisasi nilai RGB ke rentang 0-1
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

        # Temukan nilai maksimum dari setiap saluran warna
        max_r = np.max(r)
        max_g = np.max(g)
        max_b = np.max(b)

        # Temukan nilai minimum dari setiap saluran warna
        min_r = np.min(r)
        min_g = np.min(g)
        min_b = np.min(b)

        # Hitung Value (V)
        v = np.max([max_r, max_g, max_b])

        # Hitung Saturation (S)
        if v != 0:
            s = (v - np.min([min_r, min_g, min_b])) / v
        else:
            s = 0

        # Hitung Hue (H)
        if v == max_r:
            h = 60 * ((g - b) / (v - np.min([min_r, min_g, min_b])))
        elif v == max_g:
            h = 120 + 60 * ((b - r) / (v - np.min([min_r, min_g, min_b])))
        else:  # v == max_b
            h = 240 + 60 * ((r - g) / (v - np.min([min_r, min_g, min_b])))

        # Sesuaikan hue jika negatif
        h = np.where(h < 0, h + 360, h)

        return h, s, v

    def extract_features(self, image_paths):
        features = []
        for path in image_paths:
            r, g, b = self.get_rgb(path)
            h, s, v = self.rgb_to_hsv(r, g, b)  # Konversi RGB ke HSV
            features.append([h.mean(), s.mean(), v.mean()])
        return np.array(features)

    @staticmethod
    def get_rgb(image_path):
        img = cv2.imread(image_path)
        b, g, r = cv2.split(img)
        return r, g, b

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def get_actual_class(index):
        classes = [
            "matang","matang","matang","matang","matang","matang","matang","matang","matang","matang",
            "setengah_matang","setengah_matang","setengah_matang","setengah_matang","setengah_matang",
            "setengah_matang","setengah_matang","setengah_matang","setengah_matang","setengah_matang",
            "belum_matang","belum_matang","belum_matang","belum_matang","belum_matang","belum_matang",
            "belum_matang","belum_matang","belum_matang","setengah_matang","belum_matang",
        ]
        return classes[index]

    def predict(self, image_uji_path):
        r, g, b = self.get_rgb(image_uji_path)
        h, s, v = self.rgb_to_hsv(r, g, b)  # Konversi RGB ke HSV
        feature_uji = np.array([h.mean(), s.mean(), v.mean()])

        distances = self.euclidean_distance(feature_uji, self.data_latih_features)

        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[: int(self.k)]

        # Menyusun hasil jarak ke dalam struktur data yang sesuai
        distances_and_paths_labels = [
            (distances[i], self.image_latih_paths[i], self.data_latih_labels[i])
            for i in k_nearest_indices
        ]

        # Menyimpan 5 jarak tanpa diurutkan
        five_nearest_distances = distances_and_paths_labels[:5]

        # Menampilkan jarak terdekat sesuai nilai K
        print("\nJarak Terdekat Berdasarkan Nilai K:")
        for index, (distance, path, label) in enumerate(distances_and_paths_labels):
            file_name = os.path.basename(path)
            print(f"Jarak Data terdekat ke-{index + 1} ({file_name}): {distance} - Kelas: {label}")

        k_nearest_labels = [label for _, _, label in distances_and_paths_labels]
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]

        return most_common_label

if __name__ == "__main__":
    # Inisialisasi
    knn_Naga = HSV(k=4)

    # Menentukan label asli untuk data uji
    label_asli_data_uji = [
        "matang","matang","matang","matang","matang","matang","matang","matang","matang","matang",
        "setengah_matang","setengah_matang","setengah_matang","setengah_matang","setengah_matang",
        "setengah_matang","setengah_matang","setengah_matang","setengah_matang","belum_matang",
        "belum_matang","belum_matang","belum_matang","belum_matang","belum_matang","belum_matang",
        "belum_matang","belum_matang","setengah_matang","belum_matang",
    ]

    # 3. Menampilkan nilai HSV dan label dari semua data latih
    print("Nilai HSV dan Label Semua Data Latih Berdasarkan Kelas:")
    # Membuat dictionary untuk mengelompokkan data latih berdasarkan label
    data_latih_per_kelas = {}
    for path, label in zip(knn_Naga.image_latih_paths, knn_Naga.data_latih_labels):
        if label not in data_latih_per_kelas:
            data_latih_per_kelas[label] = []
        data_latih_per_kelas[label].append(path)

    # Menampilkan nilai HSV dan label untuk setiap kelompok
    for label, paths in data_latih_per_kelas.items():
        for i, path in enumerate(paths, start=1):
            r, g, b = knn_Naga.get_rgb(path)
            # Menggunakan nilai rata-rata
            h, s, v = knn_Naga.rgb_to_hsv(r.mean(), g.mean(), b.mean())
            print(f"Data Latih ke-{i}: {os.path.basename(path)} - HSV ({h.mean():.2f}, {s.mean():.2f}, {v.mean():.2f}) - Label : {label}")
        print()  # Baris kosong sebagai pemisah antar kelompok

    # Data uji
    data_uji_dir = r"D:\skripsi jilid 2\sukriadi\data_uji"
    data_uji_files = sorted(os.listdir(data_uji_dir),key=lambda x: int(os.path.splitext(x)[0].split(".")[0]),)

    predictions = []
    for i, file_name in enumerate(data_uji_files, start=1):
        image_uji_path = os.path.join(data_uji_dir, file_name)
        print(f"\nData uji {file_name}:")
        # Hitung nilai HSV untuk data uji
        print(f"HSV Data Uji {file_name} = ", end="")
        r, g, b = knn_Naga.get_rgb(image_uji_path)
        h, s, v = knn_Naga.rgb_to_hsv(r.mean(), g.mean(), b.mean())
        print(f"HSV ({h.item():.2f}, {s.item():.2f}, {v.item():.2f})")

        # Predict label
        predicted_label = knn_Naga.predict(image_uji_path)

        # Append prediction
        predictions.append(predicted_label)

    # Prediksi label
    print("\nPrediksi Label:")
    for i, pred in enumerate(predictions, start=1):
        print(f"Data Uji ke-{i} = {pred}")

    # Hitung confusion matrix
    cm = np.zeros((3, 3), dtype=int)

    # Pengaturan elemen matriks kebingungan berdasarkan label yang diurutkan
    for i in range(len(label_asli_data_uji)):
        if label_asli_data_uji[i] == "belum_matang":
            if predictions[i] == "belum_matang":
                cm[0, 0] += 1
            elif predictions[i] == "setengah_matang":
                cm[1, 0] += 1
            elif predictions[i] == "matang":
                cm[2, 0] += 1
        elif label_asli_data_uji[i] == "setengah_matang":
            if predictions[i] == "belum_matang":
                cm[0, 1] += 1
            elif predictions[i] == "setengah_matang":
                cm[1, 1] += 1
            elif predictions[i] == "matang":
                cm[2, 1] += 1
        elif label_asli_data_uji[i] == "matang":
            if predictions[i] == "belum_matang":
                cm[0, 2] += 1
            elif predictions[i] == "setengah_matang":
                cm[1, 2] += 1
            elif predictions[i] == "matang":
                cm[2, 2] += 1

    # Hitung TP, FP, FN untuk setiap kelas
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    # Tampilkan TP, FP, FN untuk setiap kelas
    print("\nTrue Positives (TP) untuk Setiap Kelas:")
    for i, tp in enumerate(TP):
        print(f"Kelas {np.unique(label_asli_data_uji)[i]}: {tp}")

    print("\nFalse Positives (FP) untuk Setiap Kelas:")
    for i, fp in enumerate(FP):
        print(f"Kelas {np.unique(label_asli_data_uji)[i]}: {fp}")

    print("\nFalse Negatives (FN) untuk Setiap Kelas:")
    for i, fn in enumerate(FN):
        print(f"Kelas {np.unique(label_asli_data_uji)[i]}: {fn}")

    # Hitung akurasi
    jumlah_data_benar = np.sum(TP)
    jumlah_data_uji = len(label_asli_data_uji)
    accuracy = (jumlah_data_benar / jumlah_data_uji) * 100
    print(f"\nAkurasi: {accuracy:.2f}%")

    # Hitung presisi untuk setiap kelas
    precision_per_kelas = TP / (TP + FP) * 100
    precision_per_kelas = np.nan_to_num(precision_per_kelas)

    # Tampilkan presisi untuk setiap kelas
    print("\nPresisi untuk Setiap Kelas:")
    for i, precision_kelas in enumerate(precision_per_kelas):
        print(f"Kelas {np.unique(label_asli_data_uji)[i]}: {precision_kelas:.2f}%")

    # Hitung recall untuk setiap kelas
    recall_per_kelas = TP / (TP + FN) * 100
    recall_per_kelas = np.nan_to_num(recall_per_kelas)

    # Tampilkan recall untuk setiap kelas
    print("\nRecall untuk Setiap Kelas:")
    for i, recall_kelas in enumerate(recall_per_kelas):
        print(f"Kelas {np.unique(label_asli_data_uji)[i]}: {recall_kelas:.2f}%")

    # Hitung F1-Score untuk setiap kelas
    f1_per_kelas = (2 * (precision_per_kelas * recall_per_kelas) / (precision_per_kelas + recall_per_kelas))
    f1_per_kelas = np.nan_to_num(f1_per_kelas)

    # Tampilkan F1-Score untuk setiap kelas
    print("\nF1-Score untuk Setiap Kelas:")
    for i, f1_kelas in enumerate(f1_per_kelas):
        print(f"Kelas {np.unique(label_asli_data_uji)[i]}: {f1_kelas:.2f}")

    # Hitung rata-rata F1-Score
    f1_mean = np.mean(f1_per_kelas)
    print(f"\nRata-Rata F1-Score: {f1_mean:.2f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", xticklabels=np.unique(label_asli_data_uji), yticklabels=np.unique(label_asli_data_uji))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
