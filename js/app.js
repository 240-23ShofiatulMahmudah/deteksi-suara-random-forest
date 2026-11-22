
// ============================================================
// APLIKASI UTAMA DETEKSI SUARA KEBOCORAN
// ============================================================

// Elemen HTML
const micStatus = document.getElementById("micStatus");
const waveCanvas = document.getElementById("waveCanvas");
const wavePlaceholder = document.getElementById("wavePlaceholder");
const recordStatus = document.getElementById("recordStatus");
const recordTimer = document.getElementById("recordTimer");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const processStatus = document.getElementById("processStatus");
const resultDisplay = document.getElementById("resultDisplay");

const bocorValue = document.getElementById("bocorValue");
const bocorBar = document.getElementById("bocorBar");
const normalValue = document.getElementById("normalValue");
const normalBar = document.getElementById("normalBar");

const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const historyBody = document.getElementById("historyBody");

// ============================================================
// KONFIGURASI APLIKASI
// ============================================================

const TARGET_SAMPLE_RATE = 16000;
const DURASI_SEGMENT = 1;
const JUMLAH_SAMPEL = TARGET_SAMPLE_RATE * DURASI_SEGMENT;

// ============================================================
// VARIABEL APLIKASI
// ============================================================

let audioContext = null;
let mediaStream = null;
let sumberAudio = null;
let audioProcessor = null;
let gainNode = null;

let sedangMerekam = false;
let modelSiap = false;

let bufferMasuk = [];
let jumlahBuffer = 0;

let antrianProses = Promise.resolve();

let waktuMulai = 0;
let animationId = null;

let audioTerakhir = null;
let samplingRateRekaman = null;

// ============================================================
// FUNGSI STATUS
// ============================================================

function ubahStatusMic(teks) {
    micStatus.textContent = teks;
}

function ubahStatusRekaman(teks) {
    recordStatus.textContent = teks;
}

function ubahStatusProses(teks) {
    processStatus.textContent = teks;
}

// ============================================================
// MEMUAT KONFIGURASI DAN MODEL
// ============================================================

async function inisialisasiAplikasi() {
    try {
        ubahStatusMic("Memuat sistem...");
        ubahStatusProses("Memuat konfigurasi audio dan model...");

        await Promise.all([
            muatKonfigurasiAudio(),
            muatModelRandomForest()
        ]);

        modelSiap = true;

        ubahStatusMic("Sistem siap");
        ubahStatusProses("Model Random Forest berhasil dimuat.");

        startBtn.disabled = false;
        stopBtn.disabled = true;

        console.log("Konfigurasi audio berhasil dimuat.");
        console.log("Model Random Forest berhasil dimuat.");

    } catch (error) {
        modelSiap = false;

        ubahStatusMic("Sistem bermasalah");
        ubahStatusProses("Gagal memuat sistem: " + error.message);

        startBtn.disabled = true;

        console.error("Kesalahan inisialisasi:", error);
    }
}

// ============================================================
// RESAMPLING AUDIO KE 16000 Hz
// ============================================================

async function resamplingAudio(audio, samplingRateAsal) {

    if (samplingRateAsal === TARGET_SAMPLE_RATE) {
        return Float32Array.from(audio);
    }

    const OfflineContext =
        window.OfflineAudioContext ||
        window.webkitOfflineAudioContext;

    if (!OfflineContext) {
        throw new Error(
            "Browser tidak mendukung OfflineAudioContext."
        );
    }

    const panjangOutput = Math.round(
        audio.length * TARGET_SAMPLE_RATE / samplingRateAsal
    );

    const offlineContext = new OfflineContext(
        1,
        panjangOutput,
        TARGET_SAMPLE_RATE
    );

    const audioBuffer = offlineContext.createBuffer(
        1,
        audio.length,
        samplingRateAsal
    );

    audioBuffer.copyToChannel(audio, 0);

    const sumber = offlineContext.createBufferSource();
    sumber.buffer = audioBuffer;
    sumber.connect(offlineContext.destination);
    sumber.start(0);

    const hasilRendering = await offlineContext.startRendering();

    return Float32Array.from(
        hasilRendering.getChannelData(0)
    );
}

// ============================================================
// MEMPROSES SATU SEGMENT AUDIO
// ============================================================

async function prosesSegmentAudio(audioAsal) {

    try {
        ubahStatusProses("Melakukan resampling audio...");

        const audio16k = await resamplingAudio(
            audioAsal,
            samplingRateRekaman
        );

        if (audio16k.length !== JUMLAH_SAMPEL) {
            throw new Error(
                "Panjang audio tidak sesuai. Ditemukan " +
                audio16k.length + " sampel."
            );
        }

        ubahStatusProses("Melakukan preprocessing audio...");

        const audioBersih = preprocessingAudio(audio16k);

        ubahStatusProses("Mengekstraksi 50 fitur audio...");

        const fitur = ekstraksiSemuaFitur(audioBersih);

        if (fitur.length !== 50) {
            throw new Error(
                "Jumlah fitur tidak sesuai: " + fitur.length
            );
        }

        ubahStatusProses("Menjalankan prediksi Random Forest...");

        const hasilPrediksi = await prediksiRandomForest(fitur);

        tampilkanHasilPrediksi(hasilPrediksi);

    } catch (error) {
        console.error("Kesalahan pemrosesan audio:", error);

        ubahStatusProses(
            "Gagal memproses audio: " + error.message
        );
    }
}

// ============================================================
// MENAMPILKAN HASIL PREDIKSI
// ============================================================


function tampilkanHasilPrediksi(hasil) {

    const DAFTAR_KELAS = [
        "AC",
        "Desis manusia",
        "Kipas",
        "Ular"
    ];

    // Ambil probabilitas dari hasil predictor
    const sumber =
        hasil?.probabilitasPerKelas ??
        hasil?.probabilitas ??
        hasil?.probabilities ??
        hasil?.probability ??
        {};

    let probabilitas = {};

    if (Array.isArray(sumber)) {
        DAFTAR_KELAS.forEach((kelas, index) => {
            probabilitas[kelas] = Number(sumber[index] ?? 0);
        });
    } else {
        DAFTAR_KELAS.forEach(kelas => {
            const key = Object.keys(sumber).find(
                k => k.toLowerCase() === kelas.toLowerCase()
            );

            probabilitas[kelas] = Number(
                key !== undefined ? sumber[key] : 0
            );
        });
    }

    // Pastikan nilai probabilitas berupa angka valid
    DAFTAR_KELAS.forEach(kelas => {
        const nilai = probabilitas[kelas];

        probabilitas[kelas] =
            Number.isFinite(nilai) && nilai >= 0 ? nilai : 0;
    });

    // Konversi ke persentase jika masih berbentuk desimal
    const total = DAFTAR_KELAS.reduce(
        (sum, kelas) => sum + probabilitas[kelas], 0
    );

    if (total > 0 && total <= 1.000001) {
        DAFTAR_KELAS.forEach(kelas => {
            probabilitas[kelas] *= 100;
        });
    }

    // Tentukan kelas tertinggi
    const kelasPrediksi = DAFTAR_KELAS.reduce(
        (terbaik, kelas) =>
            probabilitas[kelas] > probabilitas[terbaik]
                ? kelas
                : terbaik,
        DAFTAR_KELAS[0]
    );

    const kelasDariModel =
        hasil?.kelasPrediksi ??
        hasil?.prediksi ??
        hasil?.kelas ??
        hasil?.label ??
        hasil?.predicted_class;

    const kelasAkhir =
        typeof kelasDariModel === "string" &&
        DAFTAR_KELAS.includes(kelasDariModel)
            ? kelasDariModel
            : kelasPrediksi;

    const persentase = value =>
        `${Math.max(0, Math.min(100, value)).toFixed(2)}%`;

    // Buat container jika belum tersedia
    let container = document.getElementById(
        "hasil-prediksi-empat-kelas"
    );

    if (!container) {
        container = document.createElement("section");
        container.id = "hasil-prediksi-empat-kelas";

        const induk =
            document.querySelector(
                "#hasil-prediksi, #hasilPrediksi, " +
                ".hasil-prediksi, .result-container, #result"
            ) || document.body;

        induk.appendChild(container);
    }

    // Tambahkan styling
    if (!document.getElementById("style-prediksi-empat-kelas")) {
        const style = document.createElement("style");
        style.id = "style-prediksi-empat-kelas";

        style.textContent = `
            #hasil-prediksi-empat-kelas {
                width: 100%;
                margin-top: 18px;
                padding: 18px;
                box-sizing: border-box;
                border-radius: 12px;
                background: var(--card-background, #ffffff);
                color: var(--text-color, #222222);
                border: 1px solid #dddddd;
            }

            #hasil-prediksi-empat-kelas .kelas-item {
                margin: 14px 0;
            }

            #hasil-prediksi-empat-kelas .kelas-header {
                display: flex;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 7px;
                font-size: 14px;
            }

            #hasil-prediksi-empat-kelas .progress-track {
                height: 12px;
                background: #e8e8e8;
                border-radius: 20px;
                overflow: hidden;
            }

            #hasil-prediksi-empat-kelas .progress-fill {
                height: 100%;
                background: #2785cf;
                border-radius: 20px;
                transition: width .3s ease;
            }

            #hasil-prediksi-empat-kelas .kelas-terpilih {
                font-weight: bold;
                color: #16834a;
            }
        `;

        document.head.appendChild(style);
    }

    // Tampilkan hasil empat kelas
    container.innerHTML = `
        <h3>Hasil Deteksi Suara</h3>

        <div class="hasil-kelas-utama">
            <strong>Hasil prediksi:</strong>
            <span class="kelas-terpilih">
                ${kelasAkhir}
            </span>
        </div>

        <h4>Probabilitas Setiap Kelas</h4>

        ${DAFTAR_KELAS.map(kelas => `
            <div class="kelas-item">
                <div class="kelas-header">
                    <span>${kelas}</span>
                    <strong>
                        ${persentase(probabilitas[kelas])}
                    </strong>
                </div>

                <div class="progress-track">
                    <div class="progress-fill"
                         role="progressbar"
                         aria-label="Probabilitas ${kelas}"
                         aria-valuemin="0"
                         aria-valuemax="100"
                         aria-valuenow="${Math.min(100, probabilitas[kelas])}"
                         style="width:${Math.max(0, Math.min(100, probabilitas[kelas]))}%">
                    </div>
                </div>
            </div>
        `).join("")}
    `;

    // Simpan hasil terakhir untuk integrasi riwayat
    window.hasilPrediksiTerakhir = {
        kelasPrediksi: kelasAkhir,
        probabilitas: { ...probabilitas },
        waktu: new Date().toISOString()
    };

    // Simpan hasil deteksi ke tabel riwayat
    if (typeof tambahRiwayat === "function") {
        tambahRiwayat(window.hasilPrediksiTerakhir);
    }

    return window.hasilPrediksiTerakhir;
}


// ============================================================
// MENAMBAHKAN RIWAYAT DETEKSI
// ============================================================



function tambahRiwayat(hasil) {
    if (!historyBody || !hasil) return;

    const daftarKelas = [
        "AC",
        "Desis manusia",
        "Kipas",
        "Ular"
    ];

    const kelas = hasil.kelasPrediksi || "Tidak diketahui";
    const probabilitas = hasil.probabilitas || {};

    // Hapus placeholder jika masih tersedia
    const placeholder = historyBody.querySelector(".empty-history");
    if (placeholder) {
        placeholder.closest("tr").remove();
    }

    const nomor = historyBody.querySelectorAll("tr").length + 1;

    const data = [
        nomor,
        new Date(hasil.waktu || Date.now()).toLocaleTimeString("id-ID"),
        kelas,
        ...daftarKelas.map(nama => {
            const nilai = Number(probabilitas[nama] || 0);
            return nilai.toFixed(2) + "%";
        })
    ];

    const baris = document.createElement("tr");

    data.forEach(nilai => {
        const sel = document.createElement("td");
        sel.textContent = nilai;
        baris.appendChild(sel);
    });

    historyBody.prepend(baris);

    // Perbarui nomor urut
    historyBody.querySelectorAll("tr").forEach((tr, index) => {
        if (tr.cells.length === 7) {
            tr.cells[0].textContent = index + 1;
        }
    });
}



// ============================================================
// MENGAMBIL BUFFER AUDIO PER SATU DETIK
// ============================================================

function masukkanBufferAudio(data) {

    bufferMasuk.push(Float32Array.from(data));
    jumlahBuffer += data.length;

    while (
        jumlahBuffer >=
        Math.round(audioContext.sampleRate * DURASI_SEGMENT)
    ) {
        const ukuranSegment = Math.round(
            audioContext.sampleRate * DURASI_SEGMENT
        );

        const segment = new Float32Array(ukuranSegment);

        let posisi = 0;

        while (posisi < ukuranSegment) {

            const bagianPertama = bufferMasuk[0];

            const sisa = ukuranSegment - posisi;

            if (bagianPertama.length <= sisa) {

                segment.set(bagianPertama, posisi);

                posisi += bagianPertama.length;

                bufferMasuk.shift();

            } else {

                segment.set(
                    bagianPertama.subarray(0, sisa),
                    posisi
                );

                bufferMasuk[0] =
                    bagianPertama.subarray(sisa);

                posisi += sisa;
            }
        }

        jumlahBuffer -= ukuranSegment;

        antrianProses = antrianProses.then(
            () => prosesSegmentAudio(segment)
        );
    }
}

// ============================================================
// MENAMPILKAN GELOMBANG AUDIO
// ============================================================

function gambarGelombang(audio) {

    const canvas = waveCanvas;
    const ctx = canvas.getContext("2d");

    const lebar = canvas.clientWidth;
    const tinggi = canvas.clientHeight;

    if (!lebar || !tinggi) {
        return;
    }

    const rasio = window.devicePixelRatio || 1;

    if (
        canvas.width !== Math.round(lebar * rasio) ||
        canvas.height !== Math.round(tinggi * rasio)
    ) {
        canvas.width = Math.round(lebar * rasio);
        canvas.height = Math.round(tinggi * rasio);
    }

    ctx.setTransform(rasio, 0, 0, rasio, 0, 0);
    ctx.clearRect(0, 0, lebar, tinggi);

    ctx.beginPath();

    const tengah = tinggi / 2;
    const langkah = Math.max(
        1,
        Math.floor(audio.length / lebar)
    );

    for (let x = 0; x < lebar; x++) {

        const indeks = Math.min(
            x * langkah,
            audio.length - 1
        );

        const y = tengah - audio[indeks] * tengah * 0.85;

        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }

    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 2;
    ctx.stroke();
}

// ============================================================
// TIMER REKAMAN
// ============================================================

function perbaruiTimer() {

    if (!sedangMerekam) {
        return;
    }

    const detik = Math.floor(
        (performance.now() - waktuMulai) / 1000
    );

    const menit = Math.floor(detik / 60);
    const sisaDetik = detik % 60;

    recordTimer.textContent =
        String(menit).padStart(2, "0") +
        ":" +
        String(sisaDetik).padStart(2, "0");

    animationId = requestAnimationFrame(perbaruiTimer);
}

// ============================================================
// MEMULAI REKAMAN MIKROFON
// ============================================================

async function mulaiRekaman() {

    if (!modelSiap) {
        ubahStatusProses("Model belum siap.");
        return;
    }

    if (sedangMerekam) {
        return;
    }

    try {

        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error(
                "Akses mikrofon tidak tersedia. Gunakan localhost atau HTTPS."
            );
        }

        ubahStatusMic("Meminta izin mikrofon...");
        ubahStatusProses("Menghubungkan mikrofon...");

        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });

        const AudioContextClass =
            window.AudioContext ||
            window.webkitAudioContext;

        audioContext = new AudioContextClass();

        samplingRateRekaman = audioContext.sampleRate;

        await audioContext.resume();

        sumberAudio = audioContext.createMediaStreamSource(
            mediaStream
        );

        audioProcessor = audioContext.createScriptProcessor(
            4096,
            1,
            1
        );

        gainNode = audioContext.createGain();
        gainNode.gain.value = 0;

        bufferMasuk = [];
        jumlahBuffer = 0;

        antrianProses = Promise.resolve();

        audioProcessor.onaudioprocess = function(event) {

            if (!sedangMerekam) {
                return;
            }

            const data = event.inputBuffer.getChannelData(0);

            const salinan = Float32Array.from(data);

            audioTerakhir = salinan;

            if (wavePlaceholder) {
                wavePlaceholder.style.display = "none";
            }

            masukkanBufferAudio(salinan);
        };

        sumberAudio.connect(audioProcessor);
        audioProcessor.connect(gainNode);
        gainNode.connect(audioContext.destination);

        sedangMerekam = true;
        waktuMulai = performance.now();

        ubahStatusMic("Mikrofon aktif");
        ubahStatusRekaman("Sedang merekam");
        ubahStatusProses(
            "Audio sedang dikumpulkan untuk proses deteksi."
        );

        startBtn.disabled = true;
        stopBtn.disabled = false;

        perbaruiTimer();

        console.log(
            "Sampling rate mikrofon:",
            samplingRateRekaman
        );

    } catch (error) {

        console.error("Kesalahan mikrofon:", error);

        ubahStatusMic("Mikrofon tidak tersedia");
        ubahStatusProses(error.message);

        await hentikanPerangkatAudio();

        sedangMerekam = false;

        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// ============================================================
// MENGHENTIKAN PERANGKAT AUDIO
// ============================================================

async function hentikanPerangkatAudio() {

    if (audioProcessor) {
        audioProcessor.onaudioprocess = null;
        audioProcessor.disconnect();
        audioProcessor = null;
    }

    if (sumberAudio) {
        sumberAudio.disconnect();
        sumberAudio = null;
    }

    if (gainNode) {
        gainNode.disconnect();
        gainNode = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    if (audioContext) {
        await audioContext.close();
        audioContext = null;
    }
}

// ============================================================
// MENGHENTIKAN REKAMAN
// ============================================================

async function hentikanRekaman() {

    if (!sedangMerekam) {
        return;
    }

    sedangMerekam = false;

    if (animationId !== null) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    ubahStatusMic("Mikrofon tidak aktif");
    ubahStatusRekaman("Rekaman dihentikan");

    startBtn.disabled = false;
    stopBtn.disabled = true;

    await hentikanPerangkatAudio();

    // Tunggu seluruh segment yang sudah masuk antrean selesai.
    await antrianProses;

    bufferMasuk = [];
    jumlahBuffer = 0;

    ubahStatusProses(
        "Rekaman selesai. Segment audio yang lengkap telah diproses."
    );

    console.log("Rekaman dihentikan.");
}

// ============================================================
// MEMBERSIHKAN RIWAYAT
// ============================================================


function bersihkanRiwayat() {

    if (historyBody) {
        historyBody.innerHTML = "";
    }

    if (resultDisplay) {
        resultDisplay.textContent = "BELUM ADA DETEKSI";
    }

    const daftarKelas = [
        "AC",
        "Desis manusia",
        "Kipas",
        "Ular"
    ];

    daftarKelas.forEach(kelas => {
        const id = kelas
            .toLowerCase()
            .replace(/\s+/g, "");

        const value = document.getElementById(id + "Value");
        const bar = document.getElementById(id + "Bar");

        if (value) value.textContent = "0.00%";
        if (bar) bar.style.width = "0%";
    });

    const hasilContainer = document.getElementById(
        "hasil-prediksi-empat-kelas"
    );

    if (hasilContainer) {
        hasilContainer.innerHTML = "";
    }

    window.hasilPrediksiTerakhir = null;

    console.log("Riwayat deteksi berhasil dibersihkan.");
}


// ============================================================
// EVENT LISTENER
// ============================================================

startBtn.addEventListener("click", mulaiRekaman);

stopBtn.addEventListener("click", hentikanRekaman);

clearHistoryBtn.addEventListener(
    "click",
    bersihkanRiwayat
);

// ============================================================
// INISIALISASI
// ============================================================

startBtn.disabled = true;
stopBtn.disabled = true;

inisialisasiAplikasi();



// ============================================================
// INTEGRASI KONTROL APLIKASI - FINALISASI
// ============================================================

(function integrasiKontrolFinal() {

    function pasangKontrol() {

        const kontrol = [
            {
                id: "startBtn",
                fungsi: "mulaiRekaman",
                nama: "mulai"
            },
            {
                id: "stopBtn",
                fungsi: "hentikanRekaman",
                nama: "berhenti"
            },
            {
                id: "clearHistoryBtn",
                fungsi: "bersihkanRiwayat",
                nama: "hapus riwayat"
            }
        ];

        kontrol.forEach(item => {

            const tombol = document.getElementById(item.id);
            const fungsi = window[item.fungsi];

            if (!tombol) {
                console.warn("Tombol tidak ditemukan:", item.id);
                return;
            }

            if (typeof fungsi !== "function") {
                console.warn("Fungsi tidak ditemukan:", item.fungsi);
                return;
            }

            // Hindari pemasangan berulang.
            if (tombol.dataset.finalControlBound === "true") {
                return;
            }

            // Jangan menimpa handler onclick yang sudah ada.
            if (tombol.onclick !== null) {
                tombol.dataset.finalControlBound = "true";
                return;
            }

            tombol.addEventListener("click", function(event) {
                try {
                    const hasil = fungsi.call(window, event);

                    if (hasil && typeof hasil.catch === "function") {
                        hasil.catch(error => {
                            console.error(
                                "Kesalahan kontrol " + item.nama,
                                error
                            );
                        });
                    }

                } catch(error) {
                    console.error(
                        "Kesalahan kontrol " + item.nama,
                        error
                    );
                }
            });

            tombol.dataset.finalControlBound = "true";

            console.log("Kontrol terhubung:", item.id);
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            pasangKontrol,
            { once: true }
        );
    } else {
        pasangKontrol();
    }

})();

