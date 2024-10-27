# Installation

1. **Install Python**:

2. **Install OpenCV**:
    - Download OpenCV  [opencv-3.4.11-vc14_vc15.exe](https://sourceforge.net/projects/opencvlibrary/files/3.4.11/opencv-3.4.11-vc14_vc15.exe/download). 
    - Tambah lokasi 'bin' di System Environment Variable `PATH`.

    Contoh Windows:
    ```
    C:\opencv\build\x64\vc15\bin;
    ```

3. **Install Python Dependencies**:
    - Install library dari `requirements.txt` file.

    ``` execute di terminal
    pip install -r requirements.txt
    ```

## Stage

### Stage 1: Positive Image Samples Preparation

Convert semua gambar positif menjadi 1024x1024.

```execute di terminal
python image_preprocess/image_resize.py
```

Resizes semua gambar di `images/` dan simpan di `images/1024x1024`.

### Stage 2: Train Model

Latih model dengan Algoritma KNN.

```bash
python model/train_model.py
```

### Stage 3: Test the Model

Test Model yang sudah dilatih.

```bash
python model/test_model.py
```

### Stage 4: Run the App

Jalankan protorype UI. Masukkan gambar untuk melihat label/pemilik.

```bash
python app/main.py
```