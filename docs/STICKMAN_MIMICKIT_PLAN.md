# MimicKit boks projesi

## Sıradaki çalışma planı

1. Basış hedefini ring minderine bağla; destek temas oranını ölç. Tamamlandı.
2. IK hata ve ayak kaymasını azaltırken iki taraflı dövüşü koru. İlk çözüm tamamlandı.
3. Adım hızını yumuşat; kısa yan adım, blok ve darbe tepkisini görsel kayıtta
   incele. İlk kinematik çözüm tamamlandı.
4. Tekrarlayan döngüyü kır: cooldown sırasında mesafe aç, yeniden yaklaş,
   yön değiştir ve görüntüyü 30 kare/saniye kaydet. Kinematik prototip tamamlandı.
5. Temas sürekliliğini fizik üzerinden doğrula; sahte temasları ayıkla.
6. Çok eklemli IK ve motor kontrolüne geç; kinematik kök
   düzeltmesini kademeli azalt. Residual PPO ancak fizik kontrolü
   ölçülebilir biçimde çalışırsa eklensin.

Başarı ölçütü: iki tarafın isabet ürettiği 900 adım, ring/knockdown/reset
kontrolleri, ayak basışında ölçülen kayma ve temas oranı.

## Güncel uygulama ve sınırlar

Destek ayağı prototipi: ring minderi `ground*` sensöründe görünmediği için
ayakların toplam temas kuvveti ve minder yüksekliği birlikte ölçülür.
`foot_canvas_contact` iki tarafın sağ/sol temasını raporlar. Basış anında
kalça ve diz için bir damped Jacobian IK adımı, kalan yatay sapma için kök
düzeltmesi kullanılır. Basış hedefi minder üstüne taşındı ve IK kuvvetlendirildi.
Yumuşatılmış adım fazıyla 900 adımlık testlerde destek örneklerinin
215–218/285'inde temas saptandı; mesafe değiştiren ayak çalışmasıyla başsız
testte 316/408 temas görüldü. Yönlü yan adımla 377/410 temas ve en yüksek
0.0161 m IK yatay/dikey hedef hatası ölçüldü.
Bu oran henüz
güvenilir, sürekli fiziksel destek sayılmaz; sonraki aşamada temas sürekliliği
ve kök yerine bacak çözümü iyileştirilmeli.

Son dövüş dilimi: sol jab, sağ cross ve sol kroşe arasında önceki vuruştan
farklı seçim yapılır. Stamina yeterliyken olasılıklı iki vuruşlu kombinasyon
kurulur; ikinci vuruş ilkinden farklıdır ve isabet hakkı ayrı sıfırlanır.
Saldırı durumu gerçek animasyon süresiyle biter; ardından recovery zorunludur.
Test her iki karakterde üç vuruşu ve kombinasyonun ikinci vuruşunu denetler.
Episode 60 saniye; 30 kare/saniye görüntü yakın kadraj ve
can/durum/stamina/vuruş adıyla
kaydedilir. Çalıştırma: `python tools/combat_state_smoke.py --visualize`.

- İki elli guard, sol jab, sağ cross, sol kroşe ve yumuşak guard dönüşü var.
- Cross sırasında gövde dönüşü, kroşede kısa öne eğilme var; tam ağırlık
  aktarımı henüz yok.
- BLOCK, SLIP ve BACKSTEP kararları dönüşümlü uygulanıyor.
- Adım hızı basıştan çıkarken ve basışa girerken yumuşatılıyor. Blokta
  gövde sıkışması, hit-stun sırasında kısa gövde tepkisi eklendi.
- Saldırı sonrası her karakter bağımsız hedef mesafe ve dönüş yönü seçiyor.
  Cooldown ve recovery sırasında geri açılıp yana adımlıyor, sonra menzile
  yaklaşıyor. Ring kenarında içeri dönüş korunuyor.
- Stamina saldırıda 0.25 azalır, dinlenmede saniyede 0.12 yenilenir;
  0.25 altında yeni saldırı başlamaz.
- Hareket yönüne göre ileri veya yana açılan alternatif bacak animasyonu ve
  destek bacağı için ilk IK var; fiziksel yürüyüş henüz tamamlanmadı.
- Yaklaşma ve geri çekilmede kök hızı artık 0.65 saniyelik adım fazında
  uygulanır; ayak basışında duraklar. Bu kinematik adım kaymayı azaltır,
  zemin temasını fiziksel olarak sabitlemez.
- Ayak basışında seçilen destek ayağının dünya koordinatı sabitlenir.
  `support_raw_slip` düzeltme öncesi, `support_foot_slip` düzeltme sonrası
  kaymayı raporlar. Bu kinematik düzeltmedir; motorla üretilen denge değildir.
- Boks hasarı yalnız ellerden gelir. Blokta ölüm de knockdown akışına girer.
- `tools/boxing_pose_test.py` kol ayrımını, dönüşü ve dirsek yüksekliğini denetler.
- Combat testi 900 adım, iki taraflı hasar, savunma, kontrollü hit-stun,
  ring, knockdown ve reset kontrollerini kapsar.

Demo pozu doğrudan uygulandığı için residual PPO fizik kontrolü hâlâ tamamlanmış
değildir. Bu aşamayı çalışan/öğrenilmiş fiziksel denge olarak değerlendirmeyin.
Doğal ayak teması ve residual fizik doğrulaması açık işlerdir.

## Mevcut durum

Combat ortamı Newton üzerinde iki humanoid çalıştırır. Ring, yaklaşma, saldırı
cooldown'ı, recovery, hit-stun, geri itilme, can, knockdown ve iki taraflı hit
akışı hazırdır. `tools/combat_state_smoke.py` 900 adımlık başsız doğrulama yapar.

`BoxingReferenceController` rakip yönü ve mesafeden şu niyetleri üretir:

```text
GUARD / APPROACH / JAB / RECOVER / RETREAT
```

`reference_control` açıldığında referans eklem hedefi ve residual action
hesaplanır. Combat demosu pozu doğrudan yazdığı için residual motor etkisi
henüz doğrulanmış değildir.

## Kalan aşamalar

1. Ayak-canvas temasını her basış boyunca doğrula; temas kuvveti ve ayak
   yüksekliğini ayrı raporla.
2. Kinematik lateral adımı fiziksel ayak basışına ve ağırlık aktarımına bağla;
   pivot ve ring kenarında içeri dönüşü yumuşat.
3. İki bacaklı IK ve motor hedefleriyle ağırlık aktarımı kur; kök konumu
   doğrudan yazma gereksinimini azalt.
4. Fiziksel denge ve iki taraflı hasar motor kontrolünde sürerse residual PPO
   ile zamanlama ve düzeltme eğitimi ekle.

## Doğrulama

- 900 adım combat smoke ve görsel kayıt.
- Her iki tarafta cooldown ayak çalışması >1 m ve iki dönüş yönü.
- Guard pozu görsel test.
- Sol jab / sağ cross görsel test.
- İki taraflı hasar ve knockdown.
- Ring dışına çıkmama.
- Aynı anda sürekli saldırı olmaması.
- `fighter_state`, `opponent_state`, mesafe ve cooldown bilgilerinin çıktıda
  görünmesi.

## Çalıştırma

```bash
.venv-py310/bin/python tools/combat_state_smoke.py
```

Ana dosyalar:

- `mimickit/envs/combat_env.py`
- `mimickit/envs/boxing_controller.py`
- `data/envs/combat_humanoid_env.yaml`
- `tools/combat_state_smoke.py`

## Son görünür dövüş davranışı

Güncel prototip: reference modu iki kollu guard ile üç kol yörüngesi üretir
(0: sol jab, 1: sağ cross, 2: sol kroşe). Süreler 0.9/1.2/1.0 saniye;
yumuşak interpolasyon sonunda hedef tam guard'a döner. Kayıtlı klibin kök
hareketi bu modda uygulanmaz. Saldırı menzili 0.58 metredir.
Bu hâlâ kinematik demo: residual motor komutu pozu yöneten kontrol değildir.
Cross için gövde/kalça ağırlık aktarımı ve doğal guard kalibrasyonu henüz eksiktir.

Reference control artık varsayılan açık. Saldırı dışı durumlarda karakter guard
eklem hedefinde tutulur. Saldırı seçimi sabit sırada dönmez.
`fighter_attack_variant` ile `opponent_attack_variant` alanları
hangi vuruşun çalıştığını raporlar. Saldırıdan sonra state machine recovery
ve cooldown uygular; aynı yumruk karesi her simülasyon adımında tekrarlanmaz.

Savunma katmanı eklendi: rakip saldırıdayken karakter `BLOCK` durumuna geçer,
guard pozu korunur ve gelen hasar `block_damage_scale` ile azaltılır. Tehdit
bitince karakter tekrar karar döngüsüne döner.
