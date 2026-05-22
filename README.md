# Haydovchida Kamar Aniqlash

Svetafor kameralaridan haydovchining kamar taqqan yoki taqmaganini aniqlovchi
ikki bosqichli tizim: **YOLOv8 (haydovchini topadi) → MobileNetV3 (kamarni oqiydi)**.


---

## 0. Dataset haqida (muhim)

- Bir rasmda **bir nechta haydovchi** bor, har birining kamar holati har xil.
- Shuning uchun **rasm emas, har bir haydovchini label (box) bilan** belgiladim.
- Datasetni qo'lda labellab chiqdim (YOLO format).

**Dataset haqida:** 352 rasm, 453 labels — `kamar_bor` 97 (21%), `kamar_yoq` 356 (79%).
Imbalance ≈ 3.7:1. Rasmlarning **~97% kulrang/IR (tungi rasmlar)**.

---

## Task 1 — EDA

Class taqsimoti, label o'lchamlari, kun va tun bo'linishi tahlil qilindi.
Grafiklar: `task1_eda/figures/`.

| Malumot | Qiymat |
|---|---|
| Jami label rasmlar | 453ta (kamar_bor 97ta / kamar_yoq 356ta) |
| Bosh rasmlar(label qilib bolmaydigan rasmlar) | 6ta
| Imbalance | 3.7 : 1 |
| Tungi/IR rasm | ~97% |
| Label o'lchami | kichik (umumiy kadr kengligining 4–17%) |

**Qiyin Chalg'ituvchi holatlar:**
 - ~97% rasm kulrang (tungi) — kunduzgi rangli rasm deyarli yo'q
   (atiga ~11 ta). Demak model deyarli faqat kulrang sifat o'rganadi,
   kunduzgi generalizatsiya zaif bo'lishi mumkin.
 - Old oyna yarqirashi kamar ni yashiradi.
 - Bolishi mumkin bolgan muammo validationda data kam(18ta kamar_bor rasmi), shunga  ROC-AUC shovqinli yani noisy bolib qolishi mumkin edi

**Qarorga ta'siri:** 
 - Tungi rasm kop yoruglik augmentatsiyasi muhim. 
 - CLASS imbalance uchun. YOLO da (Task 2): class og'irligini hisobga olish + har class
   uchun recall'ni alohida hisoblash.
 - WeightedRandomSampler + BCE pos_weight bilan kam class ni kuchaytirish mumkin.
 - 


---

## Task 2 — YOLO Detection (2-class)

YOLOv8s, 640px, default LR (lr0=0.01), early stopping (eng yaxshi: epoch 17).

| Metrika | Natija | Maqsad |
|---|---|---|
| mAP@0.5 | **0.837** | ≥0.80 |
| mAP@0.5:0.95 | 0.471 | ≥0.55 |
| kamar_yoq Precision | **0.887** | ≥0.85 |
| kamar_yoq Recall | **0.873** | ≥0.80 |
| FPS | **64.6** | ≥30 | 

**CONFUSION MATRIX (val, 70 rasm, 89 label):**

| Haqiqiy / Bashorat | kamar_bor | kamar_yoq | background |
|---|---|---|---|
| **kamar_bor** | 12 | 2 | 4 |
| **kamar_yoq** | 2 | 61 | 4 |
| **background** | 4 | 8 | 0 |


**Qarorlar va sabablari:**
- **YOLOv8s**: qutilar kichik, Yolov8sni kichik obyekt recall i yaxshiroq.
- **Learning rate default 0.01**: kam data + pretrained uchun default yetarli deb o'yladim. Loss barqaror tushdi, pasaytirish kerak bo'lmadi. Juda sekin organganda kotarar edim.
- **Gorizontal flip O'CHIRILDI va Task 1 dagi muommolar uchun**: O'zbekistonda haydovchi doim chap tomonda, flip (o'ng  tomonli) holat yaratadi. Lekin mosaic(=1.0) bor, chunki u kichik obyektlarni aniqlashda foydali. Rotate(5.0) ozgina rotation yetadi. hsv_v = 0.5 - chunki dataset aralash (kun va tun) bu modelni turli yoritishga  chidamli qiladi.

**False Positive vs False Negative — qaysi xavfli?**
**FP xavfliroq** = kamar taqqan haydovchini "taqmagan" deyish = **sababsiz jarima**.
Shuning uchun precision'ga e'tibor berdik. Natija: **FP=2, FN=10** — model begunoh
jarima berishdan kora qoidabuzarni otkazib yuborishni koproq maqullaydi (ozimni hulosam). 

**Va ohirida shuni aytish kerak**: kamar_bor P/R past (0.67/0.80):
   Validationda atiga 18 ta kamar_bor labels bor, bu degani kam misol bilan baho "shovqinli". Imbalance (3.7:1) ham tasir qiladi. Task 3 classifier shu class'ni
   alohida kuchaytiradi (WeightedRandomSampler + pos_weight).

---


## Task 3 — Classifier (MobileNetV3-Small bilan)

Bu task YOLO da crop qilingan rasmni olib kamar bor yoki yo'qligini o'qiydi. 
ROC:`task3_classification/figures/`.

| Metrika | Natija | Maqsad |
|---|---|---|
| ROC-AUC | 0.905 | ≥0.92 | 
| Accuracy | 0.888 | ≥0.90 | 
| kamar_yoq Precision | **0.942** | ≥0.88 |
| kamar_yoq Recall | **0.916** | ≥0.85 |
| kamar_yoq f1 score | 0.929 |
| kamar_bor Precision | 0.7 | (kam class bolgani uchun, 18ta)
| kamar_bor  Recall | 0.778 | (kam class bolgani uchun, 18ta)
| kamar_bor f1 score | 0.737 |
| Threshold (Youden-J) | **0.586** | 0.5 emas |

  CONFUSION MATRIX:
                  bashorat kamar_yoq   bashorat kamar_bor
     haqiqiy kamar_yoq :     65                 6
     haqiqiy kamar_bor :      4                14

  - Youden-J (0.586):          FP=4,  FN=6,  kamar_yoq P=0.942
  - precision-scan (0.740):  FP=7,  FN=4,  kamar_yoq P=0.905

**Qarorlar:**
- **MobileNetV3-Small**: eng yengil va tez, RTSP va edge uchun yaxshiroq. efficient Net ni ham zahira qilib qoydim(kod da bor), agar MobileNet yaxshi natija bermasa deb, lekin natija qoniqtirganday boldi.
- **Threshold 0.586** (Youden-J, 0.586 chiqdi): ROC'dan TPR−FPR maksimal nuqtasi.
- **Imbalance**: WeightedRandomSampler + pos_weight bilan kam class kuchaytirildi.

**Sabablar/havflar va yollar**
 - 1. Overfitting havfi. ROC-AUC = 0.905 maqsad dan kam. validation da kam data(18ta) bolgani uchun bolishi mumkin. Train loss ham juda pas(0.001) bu degani overfit bolish havfi bor degani. 
      - Himoyalar: Augmentation (har epochda boshqacha rasm): RandomHorizontalFlip +
     ColorJitter(0.3) + RandomRotation(8) bu bilan bir xil rasmni qayta qayta model mormaydi, yodlash qiyinlashadi. WeightedRandomSampler + pos_weight - kam class uchun(kamar_bor) sun'iy kuchaytiriladi.

---

## Task 4 — E2E Pipeline

`SeatbeltDetector` klassi: frame - YOLO - crop - classifier - smoothing - qaror.

| Metrika | Natija | Minimal | 
|---|---|---|
| GPU FPS | **29.7** | ≥25 |
| Latency/kadr | **17.1 ms** | ≤40 |
| GPU Memory | **122 MB** | ≤2GB |

**Edge case'lar (kodda yozilgan):** haydovchi yo'q deganu bu bo'sh javob degani yani process_frame haydovchi yoq deb qaytaradi va davom etadi (driver_detected =  False).        
 - 2ta odam bir kadrda, Yolov model haydovchini aniqlash uchun train qilingan, faqat haydovchini oladi. 
 - Kamera titrashi - TemporalSmoother(deque maxlen=10) ishlatildi, bitta yomon kadrni yutadi, faqat ohirgi 10 kadrning kopchiligi kamar_yoq bolsa - jarima chiqadi. precision first strategya ishlaydi.
 - Qorongi sharoitda - SeatbeltDetector(clahe=True) bilan CLAHE yoqiladi, hozirgi datasetimiz kopchiligi qorongi sharoitda train bolganligi uchun - default ochik turibdi. 

**Nega 2 bosqichli?** - Classifier kattaroq, normalizatsiya qilingan crop qirqilgan rasmlarni koradi va ingichka kamar lentasi toliq kadr da korinadi va bu aniqroq boladi.
Bir-biriga halaqit qilmasdan qayta train qilish imkoni bor, smoothing classifierda yaxshiroq ishlaydi - yolo ni ozida chiqishi qiyinroq.

---

## Task 5 — Production API + Docker

Docker build + run sinab ko'rildi (`--gpus all`), ishladi.

| Test | Natija |
|---|---|
| GET /health | ok, gpu:true, model:true |
| POST /predict/image | driver/belt/bbox, warm 63ms |

| Rasmda haydovchi yoq | 200 driver_detected False |
| Buzilgan fayl | 422 File formati notogri; JPEG/PNG kutilmoqda |
| Model Yuklanmagan | 503 Model not loaded |
| Katta fayl >10MB | 413 Judaham katta file (>10MB) |

**GPU + CPU fallback:** bir image ikkala mashinada — GPU'da `--gpus all` bilan, GPU'siz
mashinada usiz (CPU, sekinroq).


## Ishga tushirish (bitta buyruq)

```bash
# 1. repo ni clone qiling, ichiga kiring
git clone <repo-url> && cd seatbelt_detector

# 2. Docker (Dockerfile task5_api/ ichida, shuning uchun -f bilan)
docker build -t seatbelt-detector -f task5_api/Dockerfile .
docker run -p 8000:8000 --gpus all seatbelt-detector   # GPU'siz: "--gpus all" ni olib tashlang

# 3. Tekshirish (DIQQAT: "/" emas — 404 beradi):
#   http://localhost:8000/health   -> JSON holat
#   http://localhost:8000/docs     -> test UI (rasm yuklab sinash)
```

> Model fayllari (`best.pt`, `mobilenet_v3_small.pt`) repo ichida — alohida yuklash
> shart emas, `docker build` ularni avtomatik oladi.

---

## Bajarilmagan yoki yetmagan narsalar

1. **mAP@0.5:0.95 (0.47<0.55), ROC-AUC (0.905<0.92), Accuracy (0.888<0.90)** — yetmadi.
   Asosiy sabab: **data kam** (282 train rasm, validation da kamar_bor atiga 18 ta). Eng muhim metrik bu — kamar_yoq precision — maqsaddan yuqori, demak tizim sababsiz jarima berishda ehtiyot bolyapti.
2. **Overfitting xavfi**: train loss juda past edi (0.001).  Qarshi himoyalar qo'llandi
   (pretrained, augmentation, weighted sampling); val P/R yaxshi qoldi, lekin
   toliq yoq qilinmadi, menimcha modelga yetarli data kamib qolgan organish uchun.
3. **ONNX/TensorRT bonus QILINMADI** — chunki **FPS allaqachon yetarli** (YOLO 64,
   pipeline 30, latency 17ms). Tezlik muammo emas edi, shuning uchun optimizatsiya
   keraksiz.
---

## Mendagi Muhit
 - 
- Windows 11, Python 3.12, PyTorch 2.11 + CUDA 12.8
- NVIDIA RTX 4050 Laptop (6GB), Ultralytics 8.2
