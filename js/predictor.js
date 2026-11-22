// ============================================
// RANDOM FOREST PREDICTOR
// Model Final - 52 Fitur, 4 Kelas
// ============================================

let modelRandomForest = null;

// ============================================
// MEMUAT MODEL RANDOM FOREST
// ============================================

async function muatModelRandomForest() {
    try {
        const response = await fetch(
            "./model/random_forest_model_baru.json"
        );

        if (!response.ok) {
            throw new Error(
                "Gagal memuat model: " + response.status
            );
        }

        const model = await response.json();

        if (model.model_type !== "RandomForestClassifier") {
            throw new Error("Jenis model tidak sesuai.");
        }

        if (model.n_estimators !== 200) {
            throw new Error("Jumlah pohon tidak sesuai.");
        }

        if (model.n_features !== 52) {
            throw new Error("Jumlah fitur harus 52.");
        }

        const kelasDiharapkan = [
            "AC",
            "Desis manusia",
            "Kipas",
            "Ular"
        ];

        if (
            !Array.isArray(model.classes) ||
            JSON.stringify(model.classes) !==
            JSON.stringify(kelasDiharapkan)
        ) {
            throw new Error("Daftar kelas tidak sesuai.");
        }

        if (
            !Array.isArray(model.feature_names) ||
            model.feature_names.length !== 52
        ) {
            throw new Error("Daftar fitur tidak valid.");
        }

        if (
            !Array.isArray(model.trees) ||
            model.trees.length !== 200
        ) {
            throw new Error("Jumlah pohon tidak valid.");
        }

        for (const [i, pohon] of model.trees.entries()) {
            const n = pohon.node_count;

            for (const key of [
                "children_left",
                "children_right",
                "feature",
                "threshold",
                "value"
            ]) {
                if (
                    !Array.isArray(pohon[key]) ||
                    pohon[key].length !== n
                ) {
                    throw new Error(
                        `Struktur ${key} pohon ${i} tidak valid.`
                    );
                }
            }
        }

        modelRandomForest = model;

        console.log("Model final berhasil dimuat.");
        console.log("Jumlah pohon:", model.n_estimators);
        console.log("Jumlah fitur:", model.n_features);
        console.log("Kelas:", model.classes);

        return true;

    } catch (error) {
        modelRandomForest = null;
        console.error("Kesalahan pemuatan model:", error);
        throw error;
    }
}


// ============================================
// PREDIKSI SATU POHON
// ============================================

function prediksiPohon(pohon, fitur) {
    let indeks = 0;

    while (true) {
        if (
            indeks < 0 ||
            indeks >= pohon.node_count
        ) {
            throw new Error("Indeks node pohon tidak valid.");
        }

        const fiturNode = pohon.feature[indeks];

        // Leaf node
        if (fiturNode === -2) {
            const nilai = pohon.value[indeks][0];

            const total = nilai.reduce(
                (a, b) => a + b,
                0
            );

            if (total <= 0) {
                throw new Error("Nilai leaf tidak valid.");
            }

            return nilai.map(v => v / total);
        }

        const nilaiFitur = fitur[fiturNode];

        if (nilaiFitur <= pohon.threshold[indeks]) {
            indeks = pohon.children_left[indeks];
        } else {
            indeks = pohon.children_right[indeks];
        }
    }
}


// ============================================
// PREDIKSI RANDOM FOREST
// ============================================

function prediksiRandomForest(fitur) {
    if (!modelRandomForest) {
        throw new Error(
            "Model belum dimuat. Jalankan " +
            "muatModelRandomForest() terlebih dahulu."
        );
    }

    if (
        !Array.isArray(fitur) &&
        !ArrayBuffer.isView(fitur)
    ) {
        throw new Error("Fitur harus berupa array numerik.");
    }

    if (fitur.length !== 52) {
        throw new Error(
            "Jumlah fitur harus 52, diterima: " + fitur.length
        );
    }

    for (let i = 0; i < fitur.length; i++) {
        if (
            typeof fitur[i] !== "number" ||
            !Number.isFinite(fitur[i])
        ) {
            throw new Error(
                "Fitur ke-" + i + " tidak valid."
            );
        }
    }

    const totalProbabilitas = new Array(4).fill(0);

    for (const pohon of modelRandomForest.trees) {
        const probabilitas = prediksiPohon(pohon, fitur);

        if (probabilitas.length !== 4) {
            throw new Error(
                "Jumlah probabilitas kelas tidak sesuai."
            );
        }

        for (let i = 0; i < 4; i++) {
            totalProbabilitas[i] += probabilitas[i];
        }
    }

    const rataRata = totalProbabilitas.map(
        nilai => nilai / modelRandomForest.trees.length
    );

    let indeksTerpilih = 0;

    for (let i = 1; i < rataRata.length; i++) {
        if (rataRata[i] > rataRata[indeksTerpilih]) {
            indeksTerpilih = i;
        }
    }

    return {
        prediksi: modelRandomForest.classes[indeksTerpilih],
        probabilitas: rataRata,

        probabilitasPerKelas: Object.fromEntries(
            modelRandomForest.classes.map(
                (kelas, i) => [kelas, rataRata[i]]
            )
        ),

        jumlahPohon: modelRandomForest.trees.length
    };
}


// ============================================
// INFORMASI MODEL
// ============================================

function getInformasiModel() {
    if (!modelRandomForest) {
        throw new Error("Model belum dimuat.");
    }

    return {
        algoritma: modelRandomForest.model_type,
        jumlahPohon: modelRandomForest.n_estimators,
        jumlahFitur: modelRandomForest.n_features,
        kelas: [...modelRandomForest.classes],
        featureNames: [...modelRandomForest.feature_names]
    };
}
