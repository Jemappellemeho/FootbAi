# FootbAi: Recherche- und Planbericht fuer den ersten Prototyp

Stand: 25.10.2026. Alle Versionen, Lizenzen und Konfigurationswerte unten wurden an diesem Tag direkt in den Repositorys (README/LICENSE/Quellcode) bzw. auf PyPI nachgesehen. Abschnitte mit **Fakt** sind belegt (Quelle angegeben), Abschnitte mit **Empfehlung** oder **Hypothese** sind meine Einschaetzung.

---

## 0. Kurzfassung

Eure Architektur ist im Kern sinnvoll. Fuenf Punkte wuerde ich aendern:

1. **Zwei Spuren statt einer Kette.** Trennt die Video-Pipeline (Spur A: Video -> Spielfeldkoordinaten) vom taktischen Vorhersagemodell (Spur B: Koordinaten -> Risiko). Spur B kann sofort mit offenen Broadcast-Trackingdaten von SkillCorner starten (10 Spiele, Meter-Koordinaten, 10 fps, inkl. Ereignissen wie "Lauf hinter die Abwehr"). Sonst haengt das gesamte Deep-Learning-Ziel an einer CV-Pipeline, die ihr erst nach Monaten zuverlaessig habt, und ihr haettet aus ein paar TV-Szenen viel zu wenige Labels.
2. **BroadTrack nicht als Baustein einplanen.** Das Repository ist gefunden (`playbox-dev/BroadTrack`), aber die Lizenz ist eine proprietaere EVS-Lizenz: nur nichtkommerzielle interne Forschung, keine Weitergabe, eure Aenderungen gehoeren EVS, Vertraulichkeitspflicht. Fuer ein Uni-Repo auf GitHub ist das ungeeignet. Ersatz: **PnLCalib** (GPL-2.0, Gewichte verfuegbar, 2026 in CVIU erschienen) oder **TVCalib** (MIT).
3. **Tracker-Vergleich auf gecachten Detektionen und gegen Ground Truth.** Erst einmal detektieren, Detektionen speichern, dann ByteTrack und BoT-SORT auf exakt denselben Boxen laufen lassen und mit HOTA/IDF1/ID-Switches gegen eine SoccerNet-Szene mit Annotationen messen. Eine beliebige TV-Szene ohne Ground Truth erlaubt nur Augenmass.
4. **Fehlender Baustein: Team- und Rollenzuordnung.** Ohne "wer gehoert zu welchem Team" und "wer ist in der Viererkette" gibt es keine Viererketten-Merkmale. Das muss als eigenes Modul in den Plan.
5. **Zeitfenster festgelegt: 7 Sekunden** (Entscheidung des Teams am 25.10.2026). Fokus des gesamten Projekts ist die **Viererkette**; Szenen gegen Dreier- oder Fuenferketten werden ausgefiltert (siehe Abschnitt Labeling).

YOLO + BoT-SORT/ByteTrack als Startpunkt bleibt richtig. Achtet aber darauf, dass Ultralytics inzwischen **TrackTrack** als Standard-Tracker nutzt; wer `tracker=` nicht explizit setzt, vergleicht etwas anderes als geplant.

---

## 1. Strategische Empfehlung: zwei Spuren

**Fakt (SkillCorner Open Data, MIT-Lizenz):** Das Repository `SkillCorner/opendata` enthaelt 10 Spiele der australischen A-League 2024/25 als Broadcast-Trackingdaten (Spieler + Ball, Meter, Ursprung Spielfeldmitte, 10 fps), plus "Dynamic Events" und "Phases of Play". Die Trackingdateien liegen per Git LFS vor (ca. 90 MB pro Spiel). Quelle: https://github.com/SkillCorner/opendata

**Fakt (selbst ausgewertet, Spiel 2017461):** Die Dynamic-Events-Datei hat 4142 Zeilen. Darunter 29 Off-Ball-Runs mit Subtyp `behind` (Lauf hinter die Abwehr). Zu jedem Lauf gibt es u. a. die Spalten `break_defensive_line` (6 von 29 wahr), `dangerous` (23 von 29), `received` (6), `lead_to_shot` (7), `push_defensive_line` (14), sowie `last_defensive_line_x_start/_end`, `last_defensive_line_height_*`, `delta_to_last_defensive_line_*`. Weitere Subtypen: `run_ahead_of_the_ball`, `support`, `cross_receiver`, `dropping_off`, `coming_short`, `pulling_wide`, `overlap`, `underlap`, `pulling_half_space`.

**Empfehlung:**

- **Spur A (Computer Vision):** Video -> Detektion -> Tracking -> Kalibrierung -> Team/Rolle -> Koordinaten in Metern. Ziel: messbar gute Koordinaten auf wenigen Szenen.
- **Spur B (Taktik + Vorhersage):** Koordinaten -> Merkmale -> Labels -> Baseline/MLP/GRU. Startet sofort auf SkillCorner-Daten (Format lesbar mit `kloppy`, BSD-3).
- **Verbindung:** Wenn Spur A Koordinaten im gleichen Format liefert, wird das in Spur B trainierte Modell auf eure eigenen Szenen angewandt. Das ist zugleich ein ehrlicher Test, wie stark Trackingfehler die Vorhersage verschlechtern (passt genau zur Erkenntnis aus Paper 2).

`handoff.md` im Repo sagt schon "Prefer tracking data over raw video for the first prototype". Die Recherche bestaetigt das; jetzt gibt es dafuer eine konkrete Datenquelle mit passenden Ereignis-Labels.

**Einschraenkungen (Fakt bzw. Hypothese):** Die SkillCorner-Ereignisse sind algorithmisch erzeugt, nicht von Menschen annotiert, und es gibt kein Video dazu. Labels muessen also stichprobenartig plausibilisiert werden (z. B. Positionen im Tracking ansehen). 10 Spiele einer Liga sind wenig; Generalisierungsaussagen bleiben explorativ.

Weitere offene Trackingdaten:
- **Metrica Sports Sample Data:** 3 anonymisierte Spiele, Tracking + Events synchronisiert, 105x68 m, Koordinaten 0..1. Keine formale Lizenzdatei, README bittet um Quellenangabe. https://github.com/metrica-sports/sample-data
- **PFF FC WM 2022:** alle 64 Spiele, Broadcast-Tracking + Events, kostenlos per Formular. Lizenzbedingungen konnte ich auf der Seite nicht einsehen, bitte vor Nutzung lesen. https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset

---

## 2. Modul fuer Modul

### Modul 1: Videoverarbeitung und Testszene

**Fakt:** SoccerNet stellt Szenen mit Ground Truth bereit, die genau fuer euren Messbedarf passen:
- SoccerNet-Tracking: 200 Clips a 30 Sekunden mit Tracking-Annotationen plus eine komplette Halbzeit. Download ueber das Paket `SoccerNet` (PyPI 0.2.0, MIT): `downloadDataTask(task="tracking-2023", split=[...])`. https://github.com/SoccerNet/sn-tracking
- SoccerNet Game State Reconstruction (GSR): 200 Clips mit Spielerpositionen in Spielfeldkoordinaten, Rolle, Team, Rueckennummer (Datensatzversion mindestens 1.3 verwenden). `task="gamestate-2024"`. https://github.com/SoccerNet/sn-gamestate

**Empfehlung:**
- Testszene 1 (quantitativ): ein SoccerNet-GSR-Clip, weil er sowohl Tracking- als auch Spielfeld-Ground-Truth hat. Damit koennt ihr Tracking (HOTA/IDF1) **und** Kalibrierung (Positionsfehler in Metern) messen.
- Testszene 2 (qualitativ): eine eigene TV-Szene von 20-30 s mit weiter Kamera und ohne Schnitt, z. B. eine Situation mit Lauf hinter die Abwehr.
- Frames: `ffmpeg -i clip.mp4 -q:v 1 frames/%06d.jpg` (dieses Namensschema verwenden SoccerNet und die Kalibrierungstools).
- Szenenschnitte erkennen und Sequenzen dort trennen (z. B. mit PySceneDetect); Tracking und Kalibrierung immer pro durchgehender Einstellung.
- Nutzungsbedingungen von SoccerNet auf soccer-net.org vor dem Download lesen; TV-Material ist urheberrechtlich geschuetzt, also nicht ins Repo legen.

**Entscheidung:** behalten, aber Testszene mit Ground Truth hinzufuegen.

### Modul 2: Spielerdetektion

**Fakt:**
- Ultralytics `ultralytics` 8.4.189 (PyPI, 25.10.2026). Aktuelle Modellfamilie ist **YOLO26** (n/s/m/l/x, COCO mAP 40.9 bis 57.5); YOLO27 ist angekuendigt, aber noch nicht verfuegbar. Lizenz **AGPL-3.0**; kommerzielle Nutzung nur mit Enterprise-Lizenz. https://github.com/ultralytics/ultralytics
- **RF-DETR** (`rfdetr` 1.11.2): Paket und Modelle N/S/M/L unter **Apache-2.0**; Groessen A/F/P/XL/2XL unter PML 1.0. Laut README hoehere COCO-AP als vergleichbare Echtzeitmodelle. https://github.com/roboflow/rf-detr
- COCO-Modelle kennen nur `person` und `sports ball`; Spieler, Torwart, Schiedsrichter werden nicht unterschieden. Roboflow stellt einen Fussball-Detektionsdatensatz bereit (aus der Kaggle-DFL-Bundesliga-Shootout-Challenge abgeleitet). https://github.com/roboflow/sports

**Zu AGPL (Fakt + Einordnung):** AGPL verlangt, dass ihr den Quellcode eures gesamten Programms unter AGPL offenlegt, wenn ihr es verteilt oder als Netzdienst anbietet. Fuer ein offenes Uni-Projekt ist das in der Regel unproblematisch. Es wird relevant, sobald jemand das Ergebnis spaeter kommerziell oder closed-source nutzen will. Das solltet ihr bewusst im README festhalten. Dies ist keine Rechtsberatung.

**Empfehlung:**
- YOLO26 (z. B. `yolo26s` oder `yolo26m`) als Startpunkt behalten, aber den Detektor hinter einer schmalen Schnittstelle kapseln (Ausgabe: `supervision.Detections` bzw. eine MOT-Textdatei). Dann ist ein Wechsel auf RF-DETR eine Zeile, falls Lizenz oder Genauigkeit es erfordern.
- Frueh auf Fussball-Klassen fine-tunen (Spieler, Torwart, Schiedsrichter, Ball). Das vorab trainierte COCO-Modell reicht fuer den ersten Durchlauf, aber Schiedsrichter wuerden sonst als Spieler in die Viererkette rutschen.
- Eingabeaufloesung erhoehen (z. B. `imgsz=1280`), weil Spieler in Weitwinkel-TV-Bildern klein sind. Hypothese, die ihr auf Testszene 1 messen koennt.

**Entscheidung:** behalten (YOLO26), RF-DETR als austauschbare Alternative vormerken.

### Modul 3: Tracking

**Fakt (Ultralytics, direkt aus `ultralytics/cfg`):**
- Verfuegbar: `botsort.yaml`, `bytetrack.yaml`, `ocsort.yaml`, `deepocsort.yaml`, `fasttrack.yaml`, `tracktrack.yaml`. **Standard ist inzwischen `tracktrack.yaml`.**
- BoT-SORT: `gmc_method: sparseOptFlow` (Kamerabewegungskompensation an), `with_reid: False`, `track_buffer: 30`.
- ByteTrack: keine Kamerabewegungskompensation, kein ReID, sonst gleiche Schwellwerte (`track_high_thresh 0.25`, `track_low_thresh 0.1`, `new_track_thresh 0.25`, `match_thresh 0.8`).

**Fakt (Roboflow `trackers` 2.6.1, Apache-2.0):** Bibliothek mit eigenstaendigen Implementierungen von SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU und McByte, alle mit derselben Schnittstelle `update(detections, frame)`, detektorunabhaengig, mit eingebauter Evaluation (`trackers eval`: CLEAR, HOTA, Identity). Die README nennt fuer SoccerNet (mit Ground-Truth-Boxen als Detektionen, also ohne Detektorfehler) HOTA 84.0 fuer ByteTrack und 84.5 fuer BoT-SORT. Klassennamen laut Paketquelle: `ByteTrackTracker`, `BoTSORTTracker` (mit `enable_cmc=True`, `cmc_method="sparseOptFlow"`). https://github.com/roboflow/trackers

**Fakt (weitere Optionen):** BoxMOT (viele Tracker mit ReID) ist AGPL-3.0. Die Original-Repos von BoT-SORT und ByteTrack sind MIT, aber nicht mehr gepflegt und auf MOT17 zugeschnitten. sn-gamestate verwendet StrongSORT mit BPBReID.

**Empfehlung:**
- Der Vergleich ByteTrack vs. BoT-SORT bleibt sinnvoll; die SoccerNet-Zahlen zeigen aber, dass sich beide bei perfekten Detektionen kaum unterscheiden. Der relevante Unterschied bei TV-Bildern ist die Kamerabewegungskompensation von BoT-SORT. Messt daher drei Varianten: ByteTrack, BoT-SORT ohne CMC, BoT-SORT mit CMC. So seht ihr, was die Kompensation bringt.
- Methodisch sauber: **einmal detektieren, Detektionen speichern, dann alle Tracker auf identischen Detektionen laufen lassen.** Dafuer eignet sich `trackers` besser als Ultralytics' eingebautes Tracking, weil Detektor und Tracker getrennt sind und Evaluation gleich mitkommt. Ultralytics' `model.track(..., tracker="botsort.yaml")` ist als schneller Sichttest trotzdem nuetzlich; dann `tracker=` immer explizit setzen.
- Kennzahlen: HOTA, IDF1, Anzahl ID-Switches, Fragmentierungen. Zusaetzlich manuelle Fehlerliste (Frame, Spieler, Fehlertyp) fuer 2-3 Schluesselmomente.
- Fuer Viererketten-Analyse sind kurze ID-Wechsel weniger schlimm, wenn die Positionen stimmen. Langfristig sollte die Identitaet ueber Team + Rolle + Position stabilisiert werden, nicht nur ueber Bild-IDs (Hypothese).

**Entscheidung:** behalten, Messaufbau aendern (gecachte Detektionen, Ground Truth, CMC als eigene Variable).

### Modul 4: Kamerakalibrierung

**Fakt: BroadTrack-Repository identifiziert.**
- Repo: https://github.com/playbox-dev/BroadTrack
- Paper: Magera, Hoyoux, Barnich, Van Droogenbroeck, "BroadTrack: Broadcast Camera Tracking for Soccer", WACV 2025, https://arxiv.org/abs/2412.01721. Laut Paper halbiert es den mittleren Reprojektionsfehler gegenueber bisherigen Methoden auf SoccerNet.
- Installation: Docker-Image, GPU (`docker run --gpus=all`), Eingabe ist ein Bildordner `%06d.jpg`, zusaetzlich eine Stativ-Datei (Kameraposition). Fuer unbekannte Stadien gibt es Standard-Kamerapositionen und ein Skript, um die Stativparameter aus einem ersten Lauf zu schaetzen.
- **Lizenz:** proprietaere Lizenz von EVS Broadcast Equipment SA. Erlaubt nur "noncommercial internal research"; keine Weitergabe oder Unterlizenzierung; Ableitungen und Aenderungen gehoeren EVS; Vertraulichkeitspflicht, die zeitlich unbegrenzt gilt. Methodisch ist BroadTrack fuer feste Stativkameras (Hauptkamera) gedacht.

**Fakt (Alternativen):**
- **PnLCalib** (GPL-2.0): Keypoint- und Liniendetektion plus Punkt-Linien-Optimierung, Verbesserungen 2026 (Linsenverzerrung, Gewichtung), akzeptiert in Computer Vision and Image Understanding. Vortrainierte Gewichte (SoccerNet, WC14, TSWC, WorldPose) als GitHub-Releases, Inferenz direkt auf Video: `python inference.py --weights_kp SV_kp --weights_line SV_lines --pnl_refine --input_path clip.mp4 --input_type video`. https://github.com/mguti97/PnLCalib
- **No Bells, Just Whistles (NBJW)** (GPL-2.0): Vorgaenger von PnLCalib; aktuell Standard-Kalibrierung in sn-gamestate (`nbjw_calib`). https://github.com/mguti97/No-Bells-Just-Whistles
- **TVCalib** (MIT): aelterer Ansatz, liberale Lizenz. https://github.com/mm4spa/tvcalib
- **Roboflow sports** (MIT): Spielfeld-Keypoints per Pose-Modell + Homographie, sehr einfach nachzuvollziehen; das Pose-Modell basiert auf YOLOv8 (also AGPL) und muss selbst trainiert werden (Colab-Notebook vorhanden).

**Empfehlung:**
- **PnLCalib als Hauptkandidat**: aktuell, gepflegt, Gewichte frei, GPL-2.0 ist mit einem offenen Uni-Repo vereinbar. Pro Frame kalibrieren, danach die Homographie zeitlich glaetten und Ausreisser verwerfen (z. B. wenn Reprojektionsfehler gross oder zu wenige Keypoints).
- **BroadTrack nur optional** als lokaler Vergleich, wenn ihr die Lizenz akzeptiert, ohne Code oder Aenderungen ins Repo zu legen. Ich wuerde ihn nicht in die Pipeline einbauen.
- Fuer den Einstieg reicht eine **Homographie** (Bodenebene -> Spielfeld), weil ihr nur Fusspunkte der Spieler (unterer Mittelpunkt der Box) auf den Boden projiziert. Volle Kameramodelle braucht ihr erst fuer Ball in der Luft.
- Messen: auf dem GSR-Clip den Positionsfehler in Metern gegen die Ground Truth. Das ist die wichtigste Zahl fuer Spur A, weil alle taktischen Merkmale in Metern rechnen.

**Entscheidung:** BroadTrack ersetzen durch PnLCalib (Fallback TVCalib bzw. Roboflow-Homographie).

### Komplett-Referenz: sn-gamestate / TrackLab

**Fakt:** sn-gamestate (GPL-3.0) baut auf TrackLab (MIT, PyPI 1.3.24) auf und liefert eine komplette Pipeline vom Video zur Minikarte: Detektion (`yolo_ultralytics`), ReID (`prtreid`), Tracking (`bpbreid_strong_sort`), Kalibrierung (`nbjw_calib`, alternativ `pnlcalib`/`tvcalib`), Rueckennummern (`mmocr`), Teamzuordnung (`kmeans_embeddings`), Teamseite (`mean_position`). Installation mit `uv`, Python 3.9, `mmcv==2.0.1`. Bewertet wird mit GS-HOTA (Positionstoleranz 5 m auf dem Spielfeld).

**Empfehlung:** Nicht als Basis fuer euren eigenen Code verwenden (alter Software-Stack, sehr viele Abhaengigkeiten, schwer verstaendlich fuer eine Projektarbeit). Aber einmal auf eurem GSR-Testclip laufen lassen, als **Referenzwert**: Wie gut ist ein etabliertes System auf derselben Szene? Damit koennt ihr eure eigene, schlankere Pipeline einordnen. Optional, wenn GPU-Zeit da ist.

### Neues Modul: Team- und Rollenzuordnung (fehlt im Plan)

**Fakt:** sn-gamestate clustert ReID-Embeddings mit KMeans; Roboflow sports nutzt SigLIP-Bildmerkmale + UMAP + KMeans in einer `TeamClassifier`-Klasse (`sports/common/team.py`).

**Empfehlung:**
- Schritt 1: Teamzuordnung pro Track (nicht pro Frame) per Clustering der Trikotfarbe oder SigLIP-Embeddings, Mehrheitsentscheidung ueber den Track.
- Schritt 2: Schiedsrichter und Torhueter ueber Detektorklasse ausschliessen.
- Schritt 3 (Video, Spur A): Viererkette pro Frame heuristisch bestimmen: die vier hintersten Feldspieler des verteidigenden Teams (in Spielfeldkoordinaten, Spielrichtung normiert). Mit Glaettung ueber die Zeit, damit die Zuordnung nicht flackert.
- Schritt 3 (Daten, Spur B), **Fakt** aus SkillCorner Spiel 2017461: Die `match.json` enthaelt pro Spieler eine Rolle (`player_role.acronym`, z. B. `LB`, `LCB`, `RCB`, `RB`, `CB`, `LWB`), und die Dynamic Events haben pro Ereignis die Spalte `defensive_structure` mit der Formation des verteidigenden Teams (in diesem Spiel u. a. `442` 468x, `424` 154x, `433` 82x, `4231` 71x, aber auch `541` 130x, `532` 86x, `523` 67x). **Empfehlung:** Viererkette = die Spieler mit Rolle LB, LCB, RCB, RB (bzw. CB) laut Aufstellung, abgeglichen mit den vier hintersten Feldspielern im Tracking; Wechsel und Rollentausch ueber die Zeit beruecksichtigen. Nur Situationen verwenden, in denen `defensive_structure` mit `4` beginnt. So ist das Modell klar auf die Viererkette zugeschnitten und vermischt nicht Dreier- und Fuenferketten.

### Ball

**Fakt:** Roboflow nennt Balltracking selbst als ungeloestes Kernproblem (klein, schnell). Sie stellen einen eigenen Ball-Datensatz und ein Trainings-Notebook bereit. SkillCorner liefert Balldaten (inkl. extrapolierter Positionen) mit.

**Empfehlung:** Ball im ersten Prototyp optional. Fuer Spur B aus den SkillCorner-Daten nehmen. Fuer Spur A spaeter einen eigenen Ballklassifikator mit hoeherer Aufloesung bzw. Bildkacheln (Tiling) plus Interpolation ueber kurze Luecken; nur Positionen mit hoher Konfidenz als Merkmal verwenden und "Ball unbekannt" explizit als Merkmal kodieren.

### Modul 5: Taktische Merkmale

**Fakt:** Die SkillCorner-Ereignisse enthalten bereits u. a. `last_defensive_line_x`, Hoehe der letzten Linie, Abstand eines Laeufers zur letzten Linie, Linienbrueche (`first_line_break`, `last_line_break`), `n_defensive_lines`, `defensive_structure`. `databallpy` (MIT) implementiert "Dangerous Accessible Space". `kloppy` (BSD-3) laedt SkillCorner, Metrica und weitere Formate in ein gemeinsames Modell. `mplsoccer` (MIT) zeichnet Spielfelder.

**Empfehlung:**
- Eure Merkmalsliste behalten. Ergaenzen: Abstand des schnellsten Angreifers zur Abseitslinie bzw. letzten Linie und dessen Geschwindigkeitskomponente in Richtung Tor; Kompaktheit (konvexe Huelle) der Kette; Zeit bis ein Verteidiger den Laeufer erreichen kann (einfache Laufzeitschaetzung).
- Eure selbst berechnete "Linienhoehe" gegen SkillCorners `last_defensive_line_x` pruefen. Das ist ein guter Plausibilitaetstest fuer Spur B.
- Koordinaten immer so normieren, dass das verteidigende Team nach links verteidigt.
- Geschwindigkeiten aus geglaetteten Positionen berechnen (z. B. Savitzky-Golay-Filter); rohe Frame-Differenzen verstaerken Trackingrauschen.

**Entscheidung:** behalten, um die genannten Punkte erweitern.

### Labeling und Definition "gefaehrliche Situation"

**Empfehlung (konkreter Vorschlag fuer Version 1):**
- Zeitpunkt t ist ein Stichprobenpunkt waehrend gegnerischen Ballbesitzes.
- Eingabe: Positionen und Geschwindigkeiten der vier Kettenspieler, der relevanten Angreifer und (falls vorhanden) des Balls im Fenster [t - 3 s, t].
- Nur Zeitpunkte, an denen das verteidigende Team mit Viererkette steht (`defensive_structure` beginnt mit `4`) und die vier Kettenspieler auf dem Platz sind.
- Label 1, wenn im Fenster (t, t + 7 s] ein Lauf hinter die Viererkette (`off_ball_run`, Subtyp `behind`) beginnt, der **die letzte Linie bricht** (`break_defensive_line`) **oder** angespielt wird (`received`) **oder** zu einem Schuss fuehrt (`lead_to_shot`). Sonst 0.
- H = **7 s** (festgelegt). 3 s und 5 s koennen als Sensitivitaetsanalyse berichtet werden. Hypothese: Bei 7 s passiert im Fenster viel, das zum Zeitpunkt t noch nicht angelegt ist (z. B. ein Ballgewinn); die Precision wird daher niedriger ausfallen, das ist erwartbar.
- Gegen Leckage: Stichprobenpunkte mit mindestens 1 s Abstand, Split immer nach Spielen.
- Zusaetzlich eine eigene, geometrische Definition (z. B. Angreifer mit Ball oder ballnah hinter der letzten Linie, im Zentrum, innerhalb von x Metern vom Tor) implementieren und gegen die SkillCorner-Labels abgleichen. Dann habt ihr eine Definition, die ihr auch auf eure eigenen Videodaten anwenden koennt.
- Aus dem einen ausgewerteten Spiel: 29 Laeufe hinter die Abwehr, davon 6 mit Linienbruch und 7 mit Schuss. Ueber 10 Spiele also grob einige Dutzend bis gut hundert positive Ereignisse. Rechnet mit starker Klassenungleichheit.

**Entscheidung:** Eure Grundidee behalten, Label an die SkillCorner-Ereignisse koppeln und manuell stichprobenartig pruefen.

### Modul 6: Vorhersagemodell

**Empfehlung:**
- Reihenfolge behalten, aber vor dem MLP zwei einfache Baselines ergaenzen: (a) die regelbasierte Referenz, (b) logistische Regression oder Gradient Boosting auf denselben Merkmalen. Bei wenigen hundert Positiven sind solche Modelle oft schwer zu schlagen (Hypothese); wenn das MLP sie nicht schlaegt, ist das ein ehrliches Ergebnis.
- GRU vor LSTM (weniger Parameter, bei kleinen Daten meist gleichwertig; Hypothese).
- GNN erst, wenn GRU auf Merkmalen klar besser ist als die statischen Baselines. Spieler als Knoten ist naheliegend, aber die Datenmenge ist der limitierende Faktor.
- Klassenungleichheit: gewichtete Loss-Funktion, Schwellwert auf Validierungsspielen waehlen, nicht auf Testspielen.

**Entscheidung:** behalten, Baselines ergaenzen.

### Modul 7: Visualisierung und Evaluation

**Empfehlung:**
- Aufteilung nach Spielen. Bei 10 SkillCorner-Spielen: Leave-one-match-out-Kreuzvalidierung oder fest 7/1/2 (Training/Validierung/Test). Stichproben, die zeitlich ueberlappen, nie auf Training und Test verteilen.
- Kennzahlen: Precision, Recall, F1 (wie geplant), dazu PR-AUC, Brier-Score, **Fehlalarme pro 10 Minuten Spielzeit** und **Vorwarnzeit** (wie viele Sekunden vor dem Ereignis die Vorhersage kommt). Die letzten beiden sind fuer euer Ziel ("moeglichst frueh") aussagekraeftiger als F1 allein.
- Spur A getrennt bewerten: HOTA/IDF1 fuer Tracking, Positionsfehler in Metern fuer Kalibrierung, und am Ende: wie stark faellt die Vorhersageguete, wenn man SkillCorner-Koordinaten durch eigene, verrauschte Koordinaten ersetzt (z. B. mit kuenstlichem Rauschen in Hoehe eures gemessenen Fehlers simulieren).
- Visualisierung mit `mplsoccer`: Spielfeld, Kette, Laeufer, Risiko-Zeitreihe.

**Entscheidung:** behalten, Kennzahlen ergaenzen.

---

## 3. Uebersicht: behalten oder ersetzen

| Schritt | Geplant | Empfehlung | Grund |
|---|---|---|---|
| Daten | eigenes TV-Video | **ergaenzen**: SoccerNet-Clips (GT) fuer Spur A, SkillCorner fuer Spur B | Messbarkeit, genug Labels |
| Detektion | YOLO | **behalten** (YOLO26), RF-DETR als Alternative | aktuell, einfach; AGPL bewusst akzeptieren |
| Tracking | BoT-SORT vs. ByteTrack | **behalten**, Messaufbau aendern | gleiche Detektionen, GT, CMC als Variable |
| Kalibrierung | BroadTrack | **ersetzen** durch PnLCalib | BroadTrack-Lizenz restriktiv |
| Team/Rolle | fehlt | **neu** | ohne sie keine Viererkette |
| Ball | optional | **behalten optional** | im Prototyp nicht kritisch |
| Merkmale | explizit | **behalten**, erweitern | gegen SkillCorner pruefbar |
| Labels | Lauf hinter Abwehr | **behalten**, an SkillCorner-Events koppeln, H = 7 s, nur Viererketten-Szenen | nachvollziehbar, pruefbar |
| Modelle | Regel, MLP, GRU/LSTM, GNN | **behalten**, LogReg/GBM-Baseline ergaenzen | ehrlicher Vergleich |
| Evaluation | P/R/F1, Spiel-Split | **behalten**, PR-AUC, Fehlalarme/10 min, Vorwarnzeit | passt zum Ziel "frueh" |

---

## 4. Plan fuer den ersten Prototyp (Spur A, Phase 1)

Ziel dieser Phase: Auf **einer** SoccerNet-Szene mit Ground Truth messbar vergleichen, wie gut ByteTrack und BoT-SORT mit denselben YOLO26-Detektionen tracken. Noch keine Kalibrierung, keine Taktik.

### Umgebung (verifiziert am 25.10.2026)

- Python 3.10 bis 3.12 (`trackers`, `supervision`, `rfdetr` brauchen >= 3.10).
- Pakete mit festen Versionen:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "ultralytics==8.4.189" "trackers==2.6.1" "supervision==0.31.2" "SoccerNet==0.2.0"
```

- GPU empfohlen (Colab reicht). YOLO26s auf CPU fuer 750 Frames geht, dauert aber Minuten.

### Schritte

1. **Testszene laden.** Einen SoccerNet-Clip mit Tracking-Ground-Truth herunterladen (`SoccerNetDownloader(...).downloadDataTask(task="tracking-2023", split=["test"])`) und einen Clip auswaehlen. Warum: nur mit Ground Truth ist "Tracker A ist besser" eine Messung statt eines Eindrucks.
2. **Detektion einmal laufen lassen und speichern** (`scripts/detect.py`): YOLO26 auf allen Frames, Klasse `person`, Ergebnisse als MOT-Textdatei `detections.txt`. Warum: beide Tracker bekommen garantiert dieselben Boxen; Detektorfehler und Trackerfehler werden trennbar.
3. **Detektionsguete pruefen**: Precision/Recall der Boxen gegen die Ground Truth (IoU 0.5). Warum: ist der Detektor schlecht, sagt der Trackervergleich wenig.
4. **Tracking** (`scripts/track.py --tracker bytetrack|botsort|botsort_nocmc`): gespeicherte Detektionen + Frames einlesen, mit `trackers.ByteTrackTracker` bzw. `trackers.BoTSORTTracker(enable_cmc=True/False)` tracken, Ergebnis als MOT-Datei.
5. **Evaluation** mit `trackers eval --gt_dir ... --predictions_dir ... --metrics '[CLEAR,HOTA,Identity]' --columns '[MOTA,HOTA,IDF1]'`: Tabelle mit HOTA, IDF1, MOTA, ID-Switches pro Variante.
6. **Visualisierung**: Video mit Boxen, IDs und Spuren (supervision-Annotatoren) fuer die manuelle Fehleranalyse; Fehlerliste als CSV.
7. **Gegencheck mit Ultralytics' eingebautem Tracking** (`model.track(..., tracker="botsort.yaml")` bzw. `"bytetrack.yaml"`) auf demselben Clip, um zu sehen, ob die Implementierungen sich unterscheiden. Optional.
8. **Eigene TV-Szene** durch dieselben Skripte schicken, nur qualitativ.

Danach (Phase 2): PnLCalib auf dem GSR-Clip, Fusspunkte in Meter projizieren, Positionsfehler messen. Parallel kann Spur B starten (SkillCorner laden, Merkmale, Labels, Baselines).

### Repository-Struktur (Vorschlag)

```
footbai/
  configs/            # Tracker- und Detektor-Parameter (YAML)
  scripts/
    detect.py         # Video/Frames -> detections.txt (MOT)
    track.py          # detections.txt + Frames -> tracks_<tracker>.txt
    evaluate.py       # Wrapper um trackers eval, schreibt results.csv
    visualize.py      # Overlay-Video
  data/               # nicht im Repo (.gitignore)
  results/            # Kennzahlen-Tabellen, kleine Plots
docs/
```

---

## 5. Offene Fragen an euch

1. Einverstanden mit der Zwei-Spuren-Aufteilung und SkillCorner als Datengrundlage fuer das Modell?
2. Habt ihr Zugang zu einer GPU (Uni-Cluster, Colab)?
3. Soll ich Phase 1 (Schritte 1-6 oben) im Repo implementieren?

---

## Quellen

- BroadTrack Repo und Lizenz: https://github.com/playbox-dev/BroadTrack, Paper: https://arxiv.org/abs/2412.01721
- PnLCalib: https://github.com/mguti97/PnLCalib, Paper: https://arxiv.org/abs/2404.08401
- No Bells, Just Whistles: https://github.com/mguti97/No-Bells-Just-Whistles
- TVCalib: https://github.com/mm4spa/tvcalib, Paper: https://arxiv.org/abs/2207.11709
- sn-gamestate: https://github.com/SoccerNet/sn-gamestate, Paper: https://arxiv.org/abs/2404.11335
- TrackLab: https://github.com/TrackingLaboratory/tracklab
- SoccerNet-Tracking: https://github.com/SoccerNet/sn-tracking
- SoccerNet 2025 Challenge-Ergebnisse: https://arxiv.org/abs/2508.19182
- Ultralytics (README, Tracker-Konfigurationen, Tracking-Doku): https://github.com/ultralytics/ultralytics, https://docs.ultralytics.com/modes/track/
- Roboflow trackers: https://github.com/roboflow/trackers
- RF-DETR: https://github.com/roboflow/rf-detr
- Roboflow sports: https://github.com/roboflow/sports
- SkillCorner Open Data: https://github.com/SkillCorner/opendata, Off-Ball-Runs-Artikel: https://skillcorner.com/articles/skillcorner-open-data-4-visualising-off-ball-runs
- Metrica Sample Data: https://github.com/metrica-sports/sample-data
- PFF FC WM-2022-Daten: https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset
- databallpy Dangerous Accessible Space: https://databallpy.readthedocs.io/en/latest/features/dangerous_accessible_space.html
- kloppy: https://github.com/PySport/kloppy
- PyPI-Versionen: ultralytics 8.4.189, trackers 2.6.1, supervision 0.31.2, rfdetr 1.11.2, kloppy 3.19.1, SoccerNet 0.2.0, tracklab 1.3.24, mplsoccer 1.8.1, databallpy 0.8.1, boxmot 25.0.0 (AGPL)

Nicht erneut geprueft: die beiden Papers aus eurem Projektkontext (ich habe eure Zusammenfassung uebernommen) sowie die Details der GSR-Gewinnerloesungen 2025.
