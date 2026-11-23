# Stickman Boxing — geliştirme planı

## Fizik geçişi — yürürlükteki çalışma

Önceki kinematik dövüş `c618a74` olarak `origin/main` dalına gönderildi.
Yeni duruşta ayaklar yaklaşık 26 cm önde/arkada, dizler kırık ve pelvis 3 cm
alçak. Destek bacağı IK çözümü iki Jacobian iterasyonunda kalça yan/ileri
eksenleriyle dizi birlikte kullanır.

`motor_pose` seçeneği eklem pozlarını doğrudan yazmak yerine Newton konum
motorlarına hedef verir. Bu modda pelvis yüksekliği fizik ve temasla değişir;
kökün yatay konumu ve yönü hâlâ dövüş denetleyicisi tarafından belirlenir.
Gerçek ayak konumu basış çıpasını ve kök düzeltmesini besler. Mod varsayılan
olmadan önce kökün yatay hareketi motor ve temas kuvvetleriyle üretilmeli.

Üç eksenli IK ile 900 adımlık motor denemesinde destek ayağı kayması
0.0394 m'den 0.0241 m'ye indi; iki taraflı isabet 6–2 kaldı. Temas 419/522
basış örneğinde görüldü. Görsel motor denemesi 4–3 isabet, 377/458 temas ve
0.0177 m en yüksek ayak kaymasıyla geçti. Kök konumu hâlâ doğrudan yazılıyor:
başsız testte 0.0845 m, görsel testte 0.1043 m'ye varan düzeltme var. Kökü
yalnız hızla taşıma denemesi 0.25 m üzeri kayma ürettiği için motor pilotuna
alınmadı.

Darbe tepkisinde eski `3.0 m/s` itme yaklaşık bir metre kaydırıyordu.
`HIT_STUN` itmesi `0.55 m/s`, sonraki geri adım `0.30 m/s` oldu. Kontrollü
test, darbeden sonraki ilk 0.4 saniyede toplam geri hareketin 0.35 m altında
kalmasını denetler.

Kalan fizik işleri: yatay kök düzeltmesini azaltmak, iki ayak için sürekli
temas ve ağırlık aktarımı kurmak, yumruğun kalça/diz basışına bağlanması,
darbe yönüne göre dengeli tepki. Motor kontrollü yatay adım ve denge
ölçülmeden PPO eklenmeyecek.

## Yeni hareketler ve gerçekçilik — uygulama sırası

Mevcut saldırılar: sol jab, sağ cross, sol kroşe. `BLOCK`, tek yönlü `SLIP`
ve `BACKSTEP` var. Yeni hareketler önce referans poz hedefi olarak eklenir;
aynı hedef motor modunda da çalıştırılır. Bir hareket, yalnız görünmesiyle
tamamlanmış sayılmaz: doğru menzilde isabet, savunmaya tepki, toparlanma ve
basış testi gerekir.

### 1. Duruş, pivot ve hareket temeli

- Gard sırasında küçük ağırlık aktarımı; saldırıda ön/arka ayağın görevini
  ayır. Rakibe dönüşü tek karede çevirmek yerine adımla pivot yap.
- İleri/geri ve iki yöne yan adımı aynı basış denetimine bağla. Ring kenarında
  pivotun içeri yönelmesini koru.
- Kabul: başsız ve görsel 900 adımda iki taraflı isabet sürer; her iki yöne
  hareket ve ring sınırı doğrulanır. Motor pilotunda en yüksek basış kayması
  3 cm altında, temas oranı en az %80 kalır. Kök düzeltmesi ayrıca raporlanır.

### 2. Yeni saldırılar — tek tek ekle

**Ön koşul — vuruş doğruluğu:** Mevcut isabet hesabı iki eli ve baş/gövde
hedeflerini birlikte tarıyor; mavi ve kırmızı tarafın kontrolü de farklı.
İki tarafı ortak bir isabet hesabına taşı. Seçili hareketin eli ve hedef
bölgesi dışında hasar verme. Her vuruşun aktif penceresinde en fazla bir
isabet kaydet; kombinasyonun ikinci vuruşuna yeni hak aç.

Her hareket için tek tanım tutulur: el, hedef bölgesi, menzil, animasyon
süresi, aktif pencere, stamina bedeli, recovery, izinli kombinasyonlar ve
basış/pivot hedefi. Aşağıdaki sayılar ilk kalibrasyon hedefidir; el yolu ve
temas ölçümüne göre ayarlanır.

| Sıra | Hareket | El / hedef | İlk menzil | Poz ve basış |
| --- | --- | --- | --- | --- |
| 1 | Sağ kroşe | Sağ / baş | 0.35–0.55 m | Dirsek bükülü yatay yay; arka ayak pivotu, kalça dönüşü |
| 2 | Sağ uppercut | Sağ / baş | 0.25–0.45 m | El aşağıdan çeneye yükselir; diz açılır, ağırlık ön ayağa geçer |
| 3 | Gövde jab'i | Sol / gövde | 0.45–0.65 m | Kısa seviye düşüşü; sol el yalnız torso/pelvis hedefler |

Uygulama sırası: önce sağ kroşe ve hedefli testi, sonra uppercut ve hedefli
testi, son olarak gövde jab'i ve hedefli testi. Yeni hareket için ayrı
`CombatState` gerekmez; `ATTACK` içindeki hareket kimliği ve süreleri
kullanılır. Seçilen vuruşun menziline yaklaşma kararı saldırı başlamadan
verilir. Rakibin mesafesi, gardı ve stamina uygun olmayan hareketi eler;
kalanlar arasındaki seçim ve kombinasyon sırası çeşitlenir.

Her hareketin hedefli testi şunları kanıtlar: doğru el ve hedefte isabet;
yanlış el/hedefte ve menzil dışında hasar yok; aktif pencere dışında isabet
yok; tek vuruşta çift hasar yok; gard pozuna dönüş var. Sonra iki modda
900 adımlık karşılıklı dövüş, ring/knockdown, ayak kayması ve 30 kare/saniye
görsel kayıt kontrol edilir. Yeni hareket mevcut üç yumruğun isabetini
bozmadan geçmelidir.

### 3. Savunma ve karşılık

- `SLIP` iki yöne ayrılır; duck ve gövde gardı eklenir. Kaçınma, yalnız poz
  değişimiyle değil el/baş/gövde temas sonuçlarıyla doğrulanır.
- Başarılı kaçınmadan sonra tek karşı yumruk fırsatı verilir. Savunma seçimi
  rakibin vuruş yönü ve hedefiyle ilişkilendirilir; sabit BLOCK→SLIP→BACKSTEP
  sırası kaldırılır.
- Kabul: her savunma en az bir hedefli saldırıyı boşa çıkarır; blok hasarı
  azaltır; karşılık cooldown ve stamina kurallarını aşmaz.

### 4. Fizik geçişi ve tempo

- Destek ayağına sürekli temas ve iki bacak arasında ağırlık aktarımı kur.
  Kökün yatay konumunu doğrudan yazma gereksinimini azalt; bu değişimi
  motor pilotunda ayrı ölç, başarısızsa varsayılan dövüşü koru.
- Hareketler bu temelde kararlı çalışınca kombinasyon seçimini, saldırı
  temposunu ve yorgunluk etkisini ayarla. PPO ancak motor denge ve iki
  taraflı hasar kök düzeltmesi olmadan doğrulandıktan sonra değerlendirilir.

**Yeni yumruk dilimi uygulandı:** iki taraf aynı isabet hesabını kullanıyor.
Yalnız seçilen el ve hareketin hedeflediği bölge hasar verebiliyor. Sağ kroşe
baş hedefli; 1.1 sn sürüyor, %30–58 aralığında aktif, 0.35–0.65 m menzilde,
gövde ve arka ayak pivotu içeriyor. Sağ uppercut baş hedefli; 1.05 sn sürüyor,
el önce alçalıp yükseliyor, %42–67 aralığında aktif, 0.25–0.65 m menzilde.
Gövde jab'i sol elle torso/pelvis hedefli; 0.85 sn sürüyor, %25–60 aralığında
aktif, 0.45–0.65 m menzilde. Her hareketin stamina bedeli ve toparlanma
ayarı ayrı. İlk kroşe ve uppercut menzil hedefleri canlı hareket ve el
geometrisine göre 0.65 m üst sınıra kalibre edildi. Bu hareketler kodla
tanımlı; öğrenilmiş bir politika henüz yok.

Hedefli test doğru el/hedef, yanlış el/hedef, menzil, aktif pencere,
tek isabet ve cooldown kapılarını iki taraf için geçti. 900 adımlık
varsayılan ve motor testleri karşılıklı hasar, ring, knockdown, temas ve ayak
kayması denetimlerinden geçti. Görsel kayıt:
`output/combat_humanoid/state_machine.mp4`. Sıradaki iş savunma ve karşılık
hareketlerini saldırı yönü ve hedefe bağlamak.

## Önceki çalışma planı

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
kalça ve diz için iki damped Jacobian IK adımı, kalan yatay sapma için kök
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

## Fizik geçişi için teknik işler

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
