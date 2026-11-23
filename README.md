# Stickman Boxing

İki çubuk adamın ringde karşılıklı boks yaptığı fizik ve kontrol denemesi.
Proje [MimicKit](https://github.com/xbpeng/MimicKit) kod tabanı üzerinde geliştirildi;
orijinal çatı belgeleri [UPSTREAM_README.md](UPSTREAM_README.md) dosyasında,
atıf bilgisi [UPSTREAM_CITATION.cff](UPSTREAM_CITATION.cff) dosyasında duruyor.

## Şu an çalışanlar

- Gerçek ring modeli ve ring sınırları
- İki taraflı saldırı, can, gard, darbe tepkisi ve knockdown
- Jab, cross, sol/sağ kroşe, sağ uppercut ve gövde jab'i
- Hareket başına el, hedef, menzil, isabet penceresi, stamina ve toparlanma
- Yaklaşma, geri çekilme, kısa yan adım, tek yönlü slip ve backstep
- Motor poz pilotu, destek ayağı IK ve ayak teması ölçümleri

Hareketler ve kararlar şu anda kodla tanımlı. **Eğitilmiş bir boks politikası
henüz yok.** Savunma becerilerini hedefli testlerle doğrulamak ve ardından
hareket seçimini öğrenen bir karar katmanı kurmak sıradaki işler. Ayrıntılı
durum ve plan: [docs/STICKMAN_MIMICKIT_PLAN.md](docs/STICKMAN_MIMICKIT_PLAN.md).

## Yerel test

Python 3.10, PyTorch, Newton ve CUDA kurulumundan sonra, depo kökünde:

```bash
python tools/boxing_pose_test.py
python tools/combat_punch_hit_test.py
python tools/combat_state_smoke.py
python tools/combat_state_smoke.py --motor-pose
python tools/combat_state_smoke.py --visualize
```

Görsel test `output/combat_humanoid/state_machine.mp4` üretir. Gerekli
MimicKit hareket verileri için orijinal [kurulum belgesine](UPSTREAM_README.md#installation)
bakın.

## Kaynak ve lisans

MimicKit, Xue Bin Peng ve katkıcıları tarafından geliştirilmiştir. Bu depo
MimicKit'ten türetilmiştir; orijinal proje ve makale için
[UPSTREAM_CITATION.cff](UPSTREAM_CITATION.cff) kullanılmalıdır. Kodun lisansı
[Apache 2.0](LICENSE) olarak korunur.
