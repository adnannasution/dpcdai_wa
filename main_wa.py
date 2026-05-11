import os
import re
import threading
import requests
import json
import schedule
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import create_engine, inspect as sa_inspect, Text, String, text as sa_text

# ─── 1. LOAD CONFIGURATION ────────────────────────────────────────────────────
load_dotenv()
DATABASE_URL    = os.getenv("DATABASE_URL")
DINOIKI_API_KEY = os.getenv("DINOIKI_API_KEY")
FONNTE_TOKEN    = os.getenv("FONNTE_TOKEN")
PRISMA_URL      = os.getenv("PRISMA_URL", "")
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY", "")
PRISMA_HEADERS  = {"x-chatbot-key": CHATBOT_API_KEY}

# ─── NEO4J CONFIG ─────────────────────────────────────────────────────────────
NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://viaduct.proxy.rlwy.net:22569")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "Neo4j@2024")

ALLOWED_NUMBERS_RAW = os.getenv("ALLOWED_NUMBERS", "")
ALLOWED_NUMBERS = [n.strip() for n in ALLOWED_NUMBERS_RAW.split(",") if n.strip()]

# ─── 2. SETUP AI ENGINE ───────────────────────────────────────────────────────
db_engine  = SQLDatabase.from_uri(DATABASE_URL, sample_rows_in_table_info=0)
_sa_engine = create_engine(DATABASE_URL)   # raw SQLAlchemy engine — untuk schema scan

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=DINOIKI_API_KEY,
    base_url="https://ai.dinoiki.com/v1",
    temperature=0.7,
)

# ─── 3. PRISMA INTEGRATION ────────────────────────────────────────────────────
def fetch_prisma_schema() -> dict:
    """Fetch schema dari PRISMA saat startup."""
    if not PRISMA_URL:
        return {}
    try:
        r = requests.get(
            f"{PRISMA_URL}/chatbot/schema",
            headers=PRISMA_HEADERS,
            timeout=15
        )
        return r.json()
    except Exception as e:
        print(f"[PRISMA] Gagal fetch schema: {e}")
        return {}

def build_prisma_schema_prompt(schema: dict) -> str:
    if not schema or "tables" not in schema:
        return ""
    lines = [
        "TABEL EKSTERNAL PRISMA TA-ex (data procurement material Turnaround):",
        "Untuk pertanyaan tentang material TA, reservasi, PR, PO, work order turnaround — gunakan query_prisma(sql).",
        "Tabel tersedia di PRISMA (BUKAN di database lokal):",
    ]
    for tbl_name, tbl in schema.get("tables", {}).items():
        col_names = tbl.get("column_names", [])
        desc      = tbl.get("description", "")
        cols_display = []
        for c in col_names:
            if c == "order":
                cols_display.append('"order"')
            else:
                cols_display.append(c)
        lines.append(f'- {tbl_name}: {desc}')
        lines.append(f'  kolom: {", ".join(cols_display)}')
    if "join_hints" in schema:
        lines.append("")
        lines.append("JOIN HINTS:")
        for k, v in schema["join_hints"].items():
            lines.append(f"  {k}: {v}")
    if "status_logic" in schema:
        lines.append("")
        lines.append("STATUS PROCUREMENT:")
        for k, v in schema["status_logic"].items():
            lines.append(f"  {k}: {v}")
    if "important_notes" in schema:
        lines.append("")
        lines.append("CATATAN PENTING:")
        for note in schema["important_notes"]:
            lines.append(f"  - {note}")
    lines += [
        "",
        "ATURAN QUERY PRISMA:",
        '- Kolom "order" WAJIB ditulis dengan tanda kutip ganda: "order"',
        "- Selalu gunakan LIMIT maksimal 50",
        "- JANGAN query tabel PRISMA ke database lokal — gunakan query_prisma(sql)",
    ]
    return "\n".join(lines)

def query_prisma(sql: str) -> dict:
    if not PRISMA_URL:
        return {"ok": False, "error": "PRISMA_URL belum dikonfigurasi"}
    try:
        r = requests.post(
            f"{PRISMA_URL}/chatbot/query",
            headers=PRISMA_HEADERS,
            json={"sql": sql},
            timeout=30
        )
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Fetch schema PRISMA saat startup
PRISMA_SCHEMA        = fetch_prisma_schema()
PRISMA_SCHEMA_PROMPT = build_prisma_schema_prompt(PRISMA_SCHEMA)
PRISMA_TABLES        = set(PRISMA_SCHEMA.get("allowed_tables", [
    "taex_reservasi", "prisma_reservasi", "kumpulan_summary",
    "sap_pr", "sap_po", "work_order"
]))

# ─── 4. DYNAMIC CATEGORICAL VALUES ────────────────────────────────────────────
_DB_SCHEMA_COLS: dict = {}   # {"table": ["col1", "col2", ...]}

_SKIP_COL_KEYWORDS = [
    "id", "no", "number", "tanggal", "date", "time", "url",
    "note", "keterangan", "alamat", "deskripsi", "description",
    "ket", "remark", "comment", "kode", "code", "path", "file",
    "nama", "name", "tag", "wbs", "pr", "po", "gr", "gi", "sa",
    "notif", "wo", "ro", "rfq", "serial"
]

def _build_db_schema_cols():
    """Scan semua tabel & kolom TEXT di DB, simpan yang layak jadi contekan."""
    global _DB_SCHEMA_COLS
    try:
        insp = sa_inspect(_sa_engine)
        tables = insp.get_table_names()
        result = {}
        with _sa_engine.connect() as conn:
            for table in tables:
                cols = insp.get_columns(table)
                text_cols = []
                for col in cols:
                    col_name = col["name"].lower()
                    if any(kw in col_name for kw in _SKIP_COL_KEYWORDS):
                        continue
                    if not isinstance(col["type"], (Text, String)):
                        continue
                    try:
                        cnt = conn.execute(
                            sa_text(f'SELECT COUNT(DISTINCT "{col["name"]}") FROM "{table}"')
                        ).scalar() or 0
                        if 1 < cnt <= 100:
                            text_cols.append(col["name"])
                    except Exception:
                        pass
                if text_cols:
                    result[table] = text_cols
        _DB_SCHEMA_COLS = result
        print(f"[schema scan] {len(_DB_SCHEMA_COLS)} tabel terindeks.")
    except Exception as e:
        print(f"[schema scan error] {e}")

def _detect_relevant_cols_sync(question: str) -> dict:
    """
    Panggil LLM kecil (sync via requests) untuk deteksi tabel & kolom kategorikal
    yang relevan dengan pertanyaan. Return: {"table": ["col1", ...]}
    """
    if not _DB_SCHEMA_COLS:
        return {}

    schema_str = "\n".join(
        f"  {tbl}: {', '.join(cols)}"
        for tbl, cols in _DB_SCHEMA_COLS.items()
    )

    detect_prompt = f"""Kamu adalah asisten yang menentukan kolom kategorikal mana yang relevan untuk sebuah pertanyaan.

Berikut daftar tabel dan kolom TEXT kategorikal yang tersedia:
{schema_str}

Pertanyaan user: "{question}"

Tentukan tabel dan kolom mana yang PALING RELEVAN dengan pertanyaan di atas.
Balas HANYA dengan JSON valid, format:
{{"tabel_nama": ["kolom1", "kolom2"], "tabel_lain": ["kolom3"]}}

Jika tidak ada yang relevan, balas: {{}}
Jangan tambahkan penjelasan apapun, hanya JSON."""

    try:
        resp = requests.post(
            f"{os.getenv('OPENAI_BASE_URL', 'https://ai.dinoiki.com/v1')}/chat/completions",
            headers={
                "Authorization": f"Bearer {DINOIKI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": detect_prompt}],
                "max_tokens": 300,
                "temperature": 0
            },
            timeout=10
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[detect_relevant_cols error] {e}")
        return {}

def _fetch_dynamic_categorical_sync(question: str) -> str:
    """
    Per pertanyaan: deteksi tabel/kolom relevan via LLM kecil (sync),
    lalu DISTINCT hanya kolom itu, return sebagai string contekan.
    """
    relevant = _detect_relevant_cols_sync(question)
    if not relevant:
        return ""

    try:
        insp     = sa_inspect(_sa_engine)
        existing = set(insp.get_table_names())
        lines    = [
            "\n\n=== NILAI KATEGORIKAL AKTUAL DI DATABASE ===",
            "Gunakan nilai-nilai berikut secara EXACT (case-sensitive) dalam SQL query:\n"
        ]

        with _sa_engine.connect() as conn:
            for table, cols in relevant.items():
                if table not in existing:
                    continue
                lines.append(f"[{table}]")
                for col in cols:
                    try:
                        result = conn.execute(
                            sa_text(
                                f'SELECT DISTINCT "{col}" FROM "{table}" '
                                f'WHERE "{col}" IS NOT NULL ORDER BY "{col}" LIMIT 50'
                            )
                        )
                        vals = [str(r[0]).strip() for r in result if r[0]]
                        if vals:
                            lines.append(f"  {col}: {' | '.join(vals)}")
                    except Exception:
                        pass
                lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"\n(Gagal load categorical values: {e})"

# Jalankan schema scan saat modul di-load
_build_db_schema_cols()

# ─── 5. SYSTEM PROMPT ─────────────────────────────────────────────────────────
CUSTOM_PROMPT = """You are a PostgreSQL expert and a helpful AI Assistant for a refinery company.
Given an input question, create a syntactically correct PostgreSQL query to run.
HANYA BERIKAN QUERY SQL MURNI, TANPA MARKDOWN ATAU BACKTICK.

Setelah mendapatkan hasil dari database, berikan jawaban akhir dalam Bahasa Indonesia yang profesional.

STRUKTUR TABEL TERSEDIA:
{table_info}

ATURAN QUERY SQL:
- Pilih tabel yang paling relevan berdasarkan nama tabel dan kolom yang tersedia.
- Jika tabel relevan kosong, jawab: "Data belum tersedia, silakan upload datanya terlebih dahulu."
- Jangan query tabel yang tidak relevan dengan pertanyaan.
- Kolom RU antar tabel mungkin berbeda format, gunakan ILIKE '%RU II%' saat JOIN.
- Selalu gunakan NULLIF(kolom_penyebut, 0) untuk menghindari division by zero.
- Gunakan ROUND(nilai::numeric, 2) untuk pembulatan.
- Jika pertanyaan melibatkan lebih dari satu tabel, gunakan JOIN yang sesuai.
- PENTING: Jangan pernah query SELECT * tanpa LIMIT. Selalu gunakan agregasi, filter, atau LIMIT 20.
- DETEKSI PERTANYAAN TIDAK PRODUKTIF: Jika user meminta salah satu dari berikut, JANGAN query — langsung tolak dengan sopan dan arahkan ke pertanyaan analisis yang lebih tepat:
  * "tampilkan semua", "list semua", "show all", "lihat semua", "ceritakan semua"
  * "tampilkan seluruh isi tabel", "dump data", "export semua"
  * Pertanyaan yang jelas akan menghasilkan ribuan baris teks panjang (action plan, progress, prokja, issue, mitigasi)
  * Pertanyaan di luar konteks maintenance kilang (cuaca, berita, pengetahuan umum, coding, dll)
  Untuk pertanyaan view massal → berikan ringkasan agregasi saja.
  Untuk pertanyaan di luar konteks → jawab: "Maaf, saya hanya dapat membantu analisis data maintenance kilang."
  PENGECUALIAN — tetap jawab dengan ramah untuk:
  * Sapaan umum (halo, hi, selamat pagi, dsb) → balas dengan ramah
  * Pertanyaan tentang kemampuan AI ini (apa yang bisa kamu lakukan, fitur apa saja, dsb) → jelaskan semua data yang tersedia
  * Ucapan terima kasih → balas dengan sopan
- Untuk icu_monitoring: kolom utama adalah ru, icu_status (Medium/High/Critical/Low), tag_no, issue, mitigation, permanent_solution, progress, target_closed, report_date. keterangan Medium = kuning, High = merah, Low = hijau.
- Untuk program_kerja_atg: kolom utama adalah refinery_unit, type, atg_eksisting, program_2024, prokja (progress), action_plan_category, target, month_update.
- Untuk paf: Plant Availability Factor — kolom type, ru, target_realisasi, value (angka PAF), plan_unplan, month.
- Untuk zero_clamp: monitoring temporary repair zero clamp — kolom ru, area, unit, tag_no_ln, type_damage, type_perbaikan, status, tanggal_dipasang, tanggal_rencana_perbaikan.
- Untuk issue_paf: daftar issue yang mempengaruhi PAF — kolom type (Primary/Secondary Unit), ru, date, issue, month_update. Kolom date bertipe DATE, filter gunakan: WHERE EXTRACT(MONTH FROM date) = 12 AND EXTRACT(YEAR FROM date) = 2024. Format month_update: W-I/W-II/W-III/W-IV + Bulan + Tahun, contoh: W-III Juli 2025. Filter bulan gunakan: WHERE month_update ILIKE '%Juli 2025%'. Filter minggu gunakan: WHERE month_update ILIKE 'W-III%Juli 2025%'.
- Untuk power_stream: status operasi equipment power & steam — kolom refinery_unit, type_equipment, equipment, status_operation, desain, kapasitas_max, average_actual.
- Untuk jumlah_eqp_utl: jumlah equipment utility per status — kolom refinery_unit, type_equipment, status_equipment, jumlah.
- Untuk critical_eqp_utl: critical equipment utility — kolom refinery_unit, type_equipment, highlight_issue, corrective_action, mitigasi_action, target_corrective, month_update. Format month_update: W-I/W-II/W-III/W-IV + Bulan + Tahun, contoh: W-I Desember 2024. Filter bulan gunakan: WHERE month_update ILIKE '%Desember 2024%'. Filter minggu gunakan: WHERE month_update ILIKE 'W-I%Desember 2024%'.
- Untuk critical_eqp_prim_sec: critical equipment primary & secondary — kolom refinery_unit, unit_proses, equipment, highlight_issue, corrective_action, mitigasi_action, month_update. Format month_update: W-I/W-II/W-III/W-IV + Bulan + Tahun, contoh: W-I Desember 2024. Filter bulan gunakan: WHERE month_update ILIKE '%Desember 2024%'. Filter minggu gunakan: WHERE month_update ILIKE 'W-I%Desember 2024%'.
- Untuk monitoring_operasi: monitoring kapasitas operasi unit proses — kolom refinery_unit, unit_proses, unit, design, minimal_capacity, plant_readiness, actual, target_sts, month_update. Format month_update: W-I/W-II/W-III/W-IV + Bulan + Tahun, contoh: W-I Desember 2024. Filter bulan gunakan: WHERE month_update ILIKE '%Desember 2024%'. Filter minggu gunakan: WHERE month_update ILIKE 'W-I%Desember 2024%'.
- Untuk inspection_plan: rencana & realisasi inspeksi equipment — kolom refinery_unit, area, tag_no_ln, type_equipment, type_inspection, due_date, plan_date, actual_date, result_remaining_life, grand_result, month_update. Format month_update: W-I/W-II/W-III/W-IV + Bulan + Tahun, contoh: W-I Desember 2024. Filter bulan gunakan: WHERE month_update ILIKE '%Desember 2024%'. Filter minggu gunakan: WHERE month_update ILIKE 'W-I%Desember 2024%'.
- Untuk tkdn: Tingkat Kandungan Dalam Negeri — kolom refinery_unit, bulan, nominal (IDR), kdn (IDR), persentase (%), tahun. Selalu tampilkan nominal dan kdn dengan format Rp dan pemisah ribuan.
- Untuk anggaran_maintenance: kolom ru, tahun, kategori, tipe, nilai_usd (USD). Selalu tampilkan nilai_usd dengan format USD dan pemisah ribuan, contoh: 1,234,567.89 USD.
- Untuk rcps_rekomendasi: rekomendasi dari RCPS — kolom kilang, rcps_no, judul_rcps, rekomendasi, traffic, pic, target, remark.
- Untuk rcps: daftar RCPS — kolom kilang, traffic, sum_of_progress, disiplin, judul_rcps, rcps_no, criticallity.
- Untuk boc: Basis of Comparison equipment — kolom ru, area, unit, equipment, status, frequency, running_hours, mttr, mtbf, hasil.
- Untuk readiness_jetty: kesiapan atau readiness operasional jetty — kolom refinery_unit, tag_no, status_operation, status_tuks, expired_tuks, status_ijin_ops, status_isps, status_struktur, status_trestle, status_mla, status_fire_protection, month_update.
- Untuk workplan_jetty: workplan perbaikan item jetty — kolom refinery_unit, tag_no, item, status_item, remark, rtl_action_plan, target, status_rtl, month_update.
- Untuk readiness_tank: kesiapan atau readiness operasional tangki — kolom refinery_unit, tag_number, type_tangki, service_tangki, prioritas, status_operational, atg_certification_validity, status_coi, status_atg, status_grounding, status_shell_course, status_roof, status_cathodic, month_update.
- Untuk workplan_tank: workplan perbaikan tangki — kolom unit, tag_no, item, remark, rtl_action_plan, target, status_rtl, month_update.
- Untuk readiness_spm: kesiapan atau readiness operasional SPM — kolom refinery_unit, tag_no, status_operation, status_laik_operasi, expired_laik_operasi, status_ijin_spl, status_mbc, status_lds, status_mooring_hawser, status_floating_hose, status_cathodic_spl, month_update.
- Untuk spm_workplan: workplan perbaikan SPM — kolom refinery_unit, tag_no, item, remark, rtl_action_plan, target, status_rtl, month_update.
- Untuk irkap_program: daftar program kerja IRKAP 2024. KOLOM YANG TERSEDIA (gunakan HANYA nama kolom ini, jangan tambah kolom lain): refinery_unit, disiplin, kategori_rkap, material_jasa, highlevel_planning_note, referensi_prokja_sebelumnya, no_program_kerja, equipment_tag_no, type_equipment, detail_type_equipment, program_kerja, step_plan_today, detail_step_plan_today, step_actual_today, detail_step_actual_today, status_step, start_plan, finish_plan, status_prognosa, kelompok_biaya, nilai_anggaran_idr, nilai_anggaran_usd, top_risk, asset_integrity. TIDAK ADA kolom month_update, bulan, tahun, atau year di tabel ini — jangan generate kolom tersebut. Untuk filter tahun gunakan EXTRACT(YEAR FROM start_plan::DATE). Tampilkan nilai_anggaran_idr dengan format Rp.
- Untuk irkap_actual: realisasi step pelaksanaan IRKAP. KOLOM YANG TERSEDIA (gunakan HANYA nama kolom ini): no, no_program, kategori_rkap, program_asset_integrity, refinery_unit, area, unit_process, tag_no, dasar_pengusulan, rekomendasi, program_kerja, disiplin, kategory_trigger, kelompok_sasaran_rk, kel_biaya, note, release_type, jadwal_pelaksanaan, jadwal_cost, jadwal_cash, strategy_penyelesaian, failure_impact, high_level_planning_note, referensi_prokja_sebelumnya, cost_center, cost_element, wbs_number, anggaran_idr, anggaran_usd, anggaran_equivalent_idr, probability_class, probability_likelyhood, economic_usd, health_safety, environment, ram_criticality, material_jasa, sumber_harga, actual_start1, actual_finish1, comp1, notif_no, actual_start2, actual_finish2, comp2, actual_start3, actual_finish3, comp3, wo_no, actual_start4, actual_finish4, comp4, ro_no, actual_start5, actual_finish5, comp5, actual_start6, actual_finish6, comp6, pr, actual_start7, actual_finish7, comp7, rfq, actual_start8, actual_finish8, comp8, po, actual_start9, actual_finish9, comp9, gr_no, actual_start10, actual_finish10, comp10, gi_no, actual_start11, actual_finish11, comp11, actual_start12, actual_finish12, comp12, actual_start13, actual_finish13, comp13, sa_no, actual_start14, actual_finish14, comp14, actual_start15, actual_finish15, comp15, current_step, status_step, status_prognosa. TIDAK ADA kolom month_update, bulan, tahun di tabel ini. Gunakan status_prognosa ('On Fiscal Year', 'Next Year', 'Closed') dan current_step untuk analisis progres.
- Untuk master_data_equipment: master data equipment dari SAP IH08 — berisi semua equipment yang terdaftar di sistem. KOLOM YANG TERSEDIA: criticality (A/B/C/Z — tingkat kritikal equipment), equipment (nomor equipment SAP), functional_location, maintenance_plant, location (kode RU/lokasi), cost_center, wbs_element, main_work_center, planner_group, planning_plant, catalog_profile, equipment_category, description (deskripsi teknis equipment), manufacturer, model_type, serial_number, changed_by, changed_on, created_by, created_on, technical_obj_type, manufact_serial_number, manufacturer_drawing_number, manufacturer_part_number, material, material_description, order_no, size_dimension, sort_field_ata. Contoh query: jumlah equipment per criticality, list equipment berdasarkan functional_location, cari equipment by description atau manufacturer. Untuk filter criticality gunakan: WHERE criticality = 'A'.

{prisma_schema}

ATURAN KLARIFIKASI — WAJIB DIIKUTI:
- WAJIB tanya klarifikasi jika pertanyaan tidak menyebut nama tabel/data spesifik secara eksplisit.
- Kata-kata berikut SAJA tanpa nama tabel spesifik = AMBIGU = WAJIB tanya dulu:
  "laporan", "data", "status", "berapa", "tampilkan", "jumlah", "lihat", "info"
- CATATAN: Kata "ru", "refinery unit", "kilang" BUKAN nama tabel spesifik — itu hanya
  filter/parameter. Jika pertanyaan hanya menyebut "ru" atau "refinery unit" tanpa nama
  tabel → tetap AMBIGU → WAJIB tanya klarifikasi.
- Nama tabel spesifik yang diakui: Pipeline, ATG, Metering, Rotor, ICU, Bad Actor, PAF,
  Zero Clamp, Power Stream, Anggaran, TKDN, RCPS, BOC, Readiness Jetty, Readiness Tank,
  Readiness SPM, Workplan Jetty, Workplan Tank, SPM Workplan, Inspection Plan,
  Monitoring Operasi, IRKAP, IRKAP Program, IRKAP Actual, Master Data Equipment, Equipment Master,
  master data, reservasi, PR, PO, material TA (PRISMA).
- DISAMBIGUASI kata "equipment":
  Jika user menyebut "equipment" BERSAMAAN dengan nama tabel lain (ICU, Pipeline, Bad Actor, dll)
  → gunakan tabel tersebut, bukan master_data_equipment.
  Contoh: "equipment di ICU" → query icu_monitoring.
  Jika user menyebut "equipment" atau "master data" TANPA nama tabel lain
  → gunakan master_data_equipment.
  Contoh: "berapa total equipment", "equipment criticality A" → query master_data_equipment.
- Jika tidak ada satupun nama tabel di atas disebut → STOP TOTAL,
  JANGAN BUAT SQL QUERY APAPUN, langsung balas dengan 1 kalimat santai saja.
  Contoh balasan: "Laporan apa yang kamu maksud? 😊 Pipeline, ATG, Metering, Rotor, ICU, Master Data Equipment, atau yang lain?"
- Jika terjadi error saat query → JANGAN ceritakan error teknis ke user.
  Cukup balas: "Hmm, sepertinya pertanyaannya kurang spesifik 😊 Laporan apa yang kamu maksud?"
- DILARANG mencoba query lalu cerita error ke user.
- DILARANG menulis paragraf panjang untuk klarifikasi.
- Cukup 1 kalimat tanya + contoh pilihan, selesai.

ATURAN FORMAT JAWABAN (KHUSUS WHATSAPP — NARASI SAJA):
1. JAWABAN FULL NARASI — JANGAN gunakan tabel HTML, JANGAN format [CHART], JANGAN [DOWNLOAD:key].
2. Jika hasil lebih dari 10 item, tampilkan ringkasan/highlight saja, maksimal 5-7 poin.
3. Gunakan poin-poin dengan tanda • jika data lebih dari satu.
4. Tebalkan poin penting dengan *teks* (bold WhatsApp).
5. Tambahkan emoticon relevan (🏭, 💰, 📊, ✅, ⚠️, 🔧, 🛢️, 🚨, 🔴).
6. Gunakan angka dengan format mudah dibaca (contoh: 1.234.567 atau Rp 1,2 M).
7. Akhiri dengan kalimat penutup singkat jika data panjang, contoh: "_(Menampilkan highlight, tanya lebih spesifik untuk detail)_"

Question: {input}"""

# ─── 6. MEMORY PER NOMOR WA ───────────────────────────────────────────────────
MAX_HISTORY = 10
wa_histories: dict[str, list] = {}

def get_history(number: str) -> list:
    return wa_histories.get(number, [])

def add_history(number: str, question: str, answer: str):
    history = wa_histories.get(number, [])
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    wa_histories[number] = history

def clear_history(number: str):
    wa_histories.pop(number, None)

# ─── 7. CORE FUNCTION ─────────────────────────────────────────────────────────
def run_wa(question: str, sender: str) -> str:
    history    = get_history(sender)
    table_info = db_engine.get_table_info()

    # Build prompt + dynamic categorical values
    prisma_prompt  = PRISMA_SCHEMA_PROMPT or "(PRISMA schema belum tersedia)"
    categorical_ctx = _fetch_dynamic_categorical_sync(question)
    _prompt = (CUSTOM_PROMPT
        .replace("{table_info}", table_info)
        .replace("{prisma_schema}", prisma_prompt)
        .replace("{input}", "")
        .replace("{{", "{")
        .replace("}}", "}")
    ) + categorical_ctx

    # Build messages dengan history
    messages = [{"role": "system", "content": _prompt}]
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    # ── Cek awal: Python keyword shortcut — bypass LLM untuk keyword yang pasti SPESIFIK ──
    _q_lower = question.lower()
    _SPESIFIK_KEYWORDS = [
        "pipeline", "atg", "metering", "rotor", "icu", "bad actor", "paf",
        "zero clamp", "power stream", "anggaran", "tkdn", "rcps", "boc",
        "readiness jetty", "readiness tank", "readiness spm",
        "workplan jetty", "workplan tank", "spm workplan",
        "inspection plan", "monitoring operasi",
        "irkap", "inspection", "prokja",
        "reservasi", "turnaround",
        "master data", "master data equipment", "equipment master",
        # Tambahan kata kunci bahasa Indonesia & kata follow-up
        "inspeksi", "realisasi", "bandingkan", "dibanding", "dibandingkan",
        "program kerja", "rencana inspeksi", "anggaran maintenance",
    ]
    _SAPAAN_KEYWORDS = [
        "halo", "hai", "hello", "hi ", "selamat pagi", "selamat siang",
        "selamat sore", "selamat malam", "terima kasih", "makasih", "thanks",
        "apa yang bisa", "kamu bisa apa", "kemampuan", "siapa kamu",
    ]

    if any(kw in _q_lower for kw in _SAPAAN_KEYWORDS) and not any(kw in _q_lower for kw in _SPESIFIK_KEYWORDS):
        intent = "SAPAAN"
    elif any(kw in _q_lower for kw in _SPESIFIK_KEYWORDS):
        intent = "SPESIFIK"
    else:
        # Sertakan history ke intent classifier agar bisa baca konteks follow-up
        history_context = ""
        if history:
            last_msgs = history[-4:]
            history_context = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content[:200]}"
                for m in last_msgs
            ])

        intent_check = llm.invoke([{
            "role": "user",
            "content": (
                f"Konteks percakapan sebelumnya:\n{history_context}\n\n"
                f"Klasifikasikan pertanyaan berikut ke salah satu kategori:\n"
                f"1. SAPAAN — jika sapaan, terima kasih, tanya kemampuan AI, atau obrolan umum yang tidak butuh data\n"
                f"2. SPESIFIK — jika menyebut nama tabel/data berikut secara eksplisit: "
                f"Pipeline, ATG, Metering, Rotor, ICU, Bad Actor, PAF, Zero Clamp, Power Stream, "
                f"Anggaran, TKDN, RCPS, BOC, Readiness Jetty, Readiness Tank, Readiness SPM, "
                f"Workplan Jetty, Workplan Tank, SPM Workplan, Inspection Plan, Monitoring Operasi, "
                f"IRKAP, IRKAP Program, IRKAP Actual, Master Data Equipment, master data, equipment master, "
                f"reservasi, PR, PO, material TA, turnaround\n"
                f"3. AMBIGU — jika tidak menyebut nama tabel spesifik apapun\n"
                f"CATATAN: Kata 'ru', 'refinery unit', 'kilang', 'equipment', 'laporan', 'data', "
                f"'status', 'berapa', 'jumlah', 'tampilkan' BUKAN nama tabel — jika hanya menyebut "
                f"kata-kata itu tanpa nama tabel spesifik maka AMBIGU.\n"
                f"PENTING: Jika pertanyaan adalah follow-up (pakai kata seperti 'bandingkan', "
                f"'realisasi', 'tersebut', 'itu', 'lanjut', 'vs') dan konteks sebelumnya sudah "
                f"menyebut topik spesifik → klasifikasi SPESIFIK.\n"
                f"Jawab hanya satu kata: SAPAAN, SPESIFIK, atau AMBIGU\n\nPertanyaan: {question}"
            )
        }])
        intent = intent_check.content.strip().upper()

    if "SAPAAN" in intent:
        greeting_response = llm.invoke(messages + [{"role": "user", "content": question}])
        return greeting_response.content

    if "AMBIGU" in intent:
        # Konfirmasi dinamis — LLM analisis apa yang kurang lalu tanya yang relevan
        history_context = ""
        if history:
            last_msgs = history[-4:]
            history_context = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content[:200]}"
                for m in last_msgs
            ])
        clarify = llm.invoke([{
            "role": "user",
            "content": (
                f"Riwayat percakapan:\n{history_context}\n\n"
                f"Pertanyaan user: {question}\n\n"
                f"Pertanyaan ini kurang lengkap untuk query database kilang. "
                f"Identifikasi informasi apa yang kurang (nama laporan, RU, tahun, filter, dll) "
                f"lalu buat satu kalimat tanya yang natural dan relevan dalam Bahasa Indonesia. "
                f"Jangan listing semua laporan yang ada, cukup tanyakan yang kurang saja. "
                f"Format WhatsApp, singkat dan ramah."
            )
        }])
        return clarify.content.strip()

    # ── Cek PRISMA via LLM ──
    prisma_check = llm.invoke([{
        "role": "user",
        "content": (
            f"Berdasarkan schema PRISMA TA-ex berikut:\n{PRISMA_SCHEMA_PROMPT}\n\n"
            f"PENTING: Jawab YA hanya jika pertanyaan EKSPLISIT menyebut salah satu dari: "
            f"reservasi, material TA, Purchase Request, PR, Purchase Order, PO, "
            f"work order turnaround, kertas kerja, delivery material, stock material TA. "
            f"Jika pertanyaan hanya menyebut 'laporan', 'data', 'status', 'berapa', "
            f"'tampilkan' tanpa konteks procurement/pengadaan TA → jawab TIDAK. "
            f"Apakah pertanyaan berikut berkaitan dengan data di PRISMA tersebut? "
            f"Jawab hanya YA atau TIDAK.\n\nPertanyaan: {question}"
        )
    }])
    is_prisma = "YA" in prisma_check.content.strip().upper()

    if is_prisma and PRISMA_URL:
        # ── Deteksi jalur: SEDERHANA atau KOMPLEKS ──
        SIMPLE_PATTERNS = [
            "berapa", "total", "jumlah", "rangkuman", "ringkasan",
            "summary", "status", "berapa yang", "sudah pr", "belum pr",
            "sudah po", "belum po", "complete", "partial",
        ]
        COMPLEX_PATTERNS = [
            "per equipment", "per order", "per material", "per plant",
            "nilai po", "net price", "harga", "breakdown", "detail",
            "join", "gabungkan", "bandingkan", "lebih dari", "kurang dari",
            "terbesar", "terkecil", "tertinggi", "terendah",
        ]

        q_low      = question.lower()
        is_simple  = any(p in q_low for p in SIMPLE_PATTERNS)
        is_complex = any(p in q_low for p in COMPLEX_PATTERNS)
        use_simple = is_simple and not is_complex

        if use_simple:
            # ── JALUR SEDERHANA: GET /chatbot/tracking ──
            params = {}
            if "belum pr" in q_low or "no-pr" in q_low or "no pr" in q_low:
                params["status"] = "no-pr"
            elif "pr created" in q_low or "sudah pr" in q_low:
                params["status"] = "pr-created"
            elif "po created" in q_low or "sudah po" in q_low:
                params["status"] = "po-created"
            elif "partial" in q_low or "sebagian" in q_low:
                params["status"] = "partial"
            elif "complete" in q_low or "selesai" in q_low or "lengkap" in q_low:
                params["status"] = "complete"

            if any(k in q_low for k in ["rangkuman", "ringkasan", "summary", "total", "berapa"]):
                params["summary_only"] = "true"

            params["chatbot_key"] = CHATBOT_API_KEY

            try:
                r = requests.get(f"{PRISMA_URL}/chatbot/tracking",
                                 params=params, timeout=30)
                prisma_result = r.json()
                print(f"[PRISMA SIMPLE] params: {params}")
                db_result = f"Hasil dari PRISMA TA-ex (jalur sederhana):\n{prisma_result}"
            except Exception as e:
                db_result = f"Gagal fetch PRISMA tracking: {str(e)}"

        else:
            # ── JALUR KOMPLEKS: POST /chatbot/query ──
            sql_messages = messages + [{"role": "user", "content": (
                f"Berikan HANYA query SQL PostgreSQL yang valid untuk pertanyaan berikut "
                f"menggunakan tabel PRISMA TA-ex. "
                f"Tabel tersedia: taex_reservasi, prisma_reservasi, kumpulan_summary, sap_pr, sap_po, work_order. "
                f"ATURAN WAJIB:\n"
                f"1. Kolom 'order' SELALU ditulis dengan tanda kutip ganda: \"order\"\n"
                f"2. Selalu tambahkan LIMIT 50 di akhir query\n"
                f"3. Untuk hitung yang sudah PR: WHERE pr IS NOT NULL AND pr != ''\n"
                f"4. Untuk hitung yang belum PR: WHERE pr IS NULL OR pr = ''\n"
                f"5. Untuk status PO: JOIN sap_po ON sap_po.purchreq = taex_reservasi.pr\n"
                f"6. Gunakan COUNT(*) atau COUNT(DISTINCT ...) untuk agregasi\n"
                f"7. HANYA output SQL murni, tanpa penjelasan, tanpa markdown, tanpa backtick\n"
                f"\nPertanyaan: {question}"
            )}]
            sql_response = llm.invoke(sql_messages)
            sql_query    = sql_response.content.replace("```sql", "").replace("```", "").strip()
            print(f"[PRISMA SQL] {sql_query}")

            prisma_result = query_prisma(sql_query)
            if prisma_result.get("ok"):
                rows      = prisma_result.get('rows', 0)
                data      = prisma_result.get('data', [])
                db_result = f"Hasil dari PRISMA TA-ex ({rows} baris):\n{data}"
            else:
                err = prisma_result.get('error', 'Unknown error')
                print(f"[PRISMA ERROR] SQL: {sql_query}")
                print(f"[PRISMA ERROR] Error: {err}")
                db_result = (
                    f"Query PRISMA gagal. SQL yang dicoba: {sql_query}. "
                    f"Error: {err}."
                )
    else:
        # ── LOCAL PATH ──
        sql_messages = messages + [{"role": "user", "content": (
            f"Berikan HANYA query SQL PostgreSQL yang valid untuk: {question}. "
            f"Tanpa penjelasan, tanpa markdown."
        )}]
        sql_response = llm.invoke(sql_messages)
        sql_query    = sql_response.content.replace("```sql", "").replace("```", "").strip()
        print(f"[LOCAL SQL] {sql_query}")

        try:
            db_result = db_engine.run(sql_query)
        except Exception as e:
            db_result = f"Query error: {str(e)}"

    # Generate jawaban final — format WhatsApp
    answer_messages = messages + [
        {"role": "user", "content": question},
        {"role": "user", "content": (
            f"Hasil query SQL:\n{db_result}\n\n"
            f"Berikan jawaban final dalam Bahasa Indonesia sesuai aturan format WhatsApp. "
            f"Ingat: narasi saja, tidak ada tabel HTML, tidak ada [CHART], tidak ada [DOWNLOAD:key]."
        )}
    ]
    final_response = llm.invoke(answer_messages)
    answer = final_response.content.replace("```sql", "").replace("```", "").strip()

    # Bersihkan artefak format web yang tidak relevan di WA
    answer = re.sub(r'\[CHART\].*?\[/CHART\]', '', answer, flags=re.DOTALL)
    answer = re.sub(r'<table.*?>.*?</table>', '', answer, flags=re.DOTALL)
    answer = re.sub(r'<[^>]+>', '', answer)
    answer = re.sub(r'\[DOWNLOAD:\w+\]', '', answer).strip()

    # Simpan ke history
    add_history(sender, question, answer)

    return answer

# ─── 8. HELPER ────────────────────────────────────────────────────────────────
def send_wa(target: str, message: str) -> dict:
    response = requests.post(
        "https://api.fonnte.com/send",
        headers={"Authorization": FONNTE_TOKEN},
        data={"target": target, "message": message},
        timeout=30,
    )
    return response.json()

# ─── 8b. KNOWLEDGE GRAPH SYNC ─────────────────────────────────────────────────
def run_graph_sync():
    """
    Sync master_data_equipment + boc ke Neo4j Knowledge Graph.
    - Otomatis tiap tengah malam via scheduler
    - Manual via endpoint /sync/graph?key=xxx

    Kolom hasil di BOC:
    - N+0    = tidak ada equipment standby dalam grup (KRITIS!)
    - N+1    = ada 1 equipment standby
    - N+2    = ada 2 equipment standby
    - Single = equipment tidak punya grup, berdiri sendiri
    """
    import psycopg2
    print("[GRAPH SYNC] Mulai sync knowledge graph...")
    try:
        pg  = psycopg2.connect(DATABASE_URL)
        neo = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

        # Buat index supaya query cepat
        with neo.session() as session:
            session.run("CREATE INDEX eq_id  IF NOT EXISTS FOR (e:Equipment) ON (e.equipment)")
            session.run("CREATE INDEX boc_id IF NOT EXISTS FOR (b:BOC) ON (b.equipment)")

        # ── Sync master_data_equipment ─────────────────────────────────────
        cur = pg.cursor("eq_cur")
        cur.execute("""
            SELECT equipment, criticality, functional_location,
                   maintenance_plant, location, cost_center,
                   equipment_category, description, manufacturer,
                   model_type, technical_obj_type, material,
                   material_description, size_dimension, sort_field_ata
            FROM master_data_equipment
            WHERE equipment IS NOT NULL
        """)
        total_eq = 0
        while True:
            rows = cur.fetchmany(1000)
            if not rows:
                break
            with neo.session() as session:
                session.run("""
                    UNWIND $rows AS r
                    MERGE (e:Equipment {equipment: r.equipment})
                    SET e.criticality          = r.criticality,
                        e.functional_location  = r.functional_location,
                        e.maintenance_plant    = r.maintenance_plant,
                        e.location             = r.location,
                        e.cost_center          = r.cost_center,
                        e.equipment_category   = r.equipment_category,
                        e.description          = r.description,
                        e.manufacturer         = r.manufacturer,
                        e.model_type           = r.model_type,
                        e.technical_obj_type   = r.technical_obj_type,
                        e.material             = r.material,
                        e.material_description = r.material_description,
                        e.size_dimension       = r.size_dimension,
                        e.sort_field_ata       = r.sort_field_ata
                """, rows=[{
                    "equipment":           r[0] or "",
                    "criticality":         r[1] or "",
                    "functional_location": r[2] or "",
                    "maintenance_plant":   r[3] or "",
                    "location":            r[4] or "",
                    "cost_center":         r[5] or "",
                    "equipment_category":  r[6] or "",
                    "description":         r[7] or "",
                    "manufacturer":        r[8] or "",
                    "model_type":          r[9] or "",
                    "technical_obj_type":  r[10] or "",
                    "material":            r[11] or "",
                    "material_description":r[12] or "",
                    "size_dimension":      r[13] or "",
                    "sort_field_ata":      r[14] or ""
                } for r in rows])
            total_eq += len(rows)
            print(f"[GRAPH SYNC] Equipment: {total_eq} node...", end="\r")
        cur.close()
        print(f"\n[GRAPH SYNC] ✅ Equipment selesai: {total_eq} node")

        # ── Sync BOC ───────────────────────────────────────────────────────
        cur2 = pg.cursor("boc_cur")
        cur2.execute("""
            SELECT equipment, ru, area, unit, grup_equipment,
                   status, frequency, running_hours, mttr, mtbf, hasil
            FROM boc
            WHERE equipment IS NOT NULL
        """)
        total_boc = 0
        while True:
            rows = cur2.fetchmany(1000)
            if not rows:
                break
            with neo.session() as session:
                session.run("""
                    UNWIND $rows AS r
                    MERGE (e:Equipment {equipment: r.equipment})
                    MERGE (b:BOC {equipment: r.equipment})
                    SET b.ru               = r.ru,
                        b.area             = r.area,
                        b.unit             = r.unit,
                        b.grup_equipment   = r.grup_equipment,
                        b.status           = r.status,
                        b.frequency        = r.frequency,
                        b.running_hours    = r.running_hours,
                        b.mttr             = r.mttr,
                        b.mtbf             = r.mtbf,
                        b.hasil            = r.hasil,
                        b.redundancy_status = r.redundancy_status
                    MERGE (e)-[:PUNYA_PERFORMA]->(b)
                """, rows=[{
                    "equipment":      r[0] or "",
                    "ru":             r[1] or "",
                    "area":           r[2] or "",
                    "unit":           r[3] or "",
                    "grup_equipment": r[4] or "",
                    "status":         r[5] or "",
                    "frequency":      r[6] or 0,
                    "running_hours":  float(r[7]) if r[7] else 0.0,
                    "mttr":           float(r[8]) if r[8] else 0.0,
                    "mtbf":           float(r[9]) if r[9] else 0.0,
                    "hasil":          r[10] or "",
                    "redundancy_status": (
                        "Tidak ada standby - KRITIS"  if r[10] == "N+0"    else
                        "Ada 1 equipment standby"      if r[10] == "N+1"    else
                        "Ada 2 equipment standby"      if r[10] == "N+2"    else
                        "Equipment tunggal tanpa grup" if r[10] == "Single" else
                        r[10] or ""
                    )
                } for r in rows])
            total_boc += len(rows)
            print(f"[GRAPH SYNC] BOC: {total_boc} relasi...", end="\r")
        cur2.close()
        print(f"\n[GRAPH SYNC] ✅ BOC selesai: {total_boc} relasi")

        neo.close()
        pg.close()
        print("[GRAPH SYNC] 🎉 Knowledge graph berhasil diperbarui!")
        return {"ok": True, "equipment": total_eq, "boc": total_boc}

    except Exception as e:
        print(f"[GRAPH SYNC ERROR] {e}")
        return {"ok": False, "error": str(e)}


def _start_graph_scheduler():
    """Scheduler otomatis sync graph tiap tengah malam."""
    schedule.every().day.at("00:00").do(run_graph_sync)
    print("[SCHEDULER] Graph sync dijadwalkan tiap tengah malam ✅")
    while True:
        schedule.run_pending()
        time.sleep(60)

# Jalankan scheduler di background saat app start
threading.Thread(target=_start_graph_scheduler, daemon=True).start()


# ─── 9. FLASK APP ─────────────────────────────────────────────────────────────
app = Flask(__name__)

# Set untuk deduplication — simpan message ID yang sudah diproses
processed_messages: set = set()

@app.route("/webhook", methods=["POST"])
def webhook():
    data    = request.get_json(force=True, silent=True) or {}
    sender  = data.get("sender", "")
    message = data.get("message", "").strip()

    # Deduplication — tolak pesan yang sama jika dikirim ulang Fonnte
    msg_id = data.get("id") or data.get("message_id") or f"{sender}:{message[:80]}"
    if msg_id in processed_messages:
        print(f"[DUPLIKAT DIABAIKAN] {msg_id}")
        return jsonify({"status": "duplicate"}), 200
    processed_messages.add(msg_id)

    # ── Deteksi apakah pesan dari grup ──
    is_group    = data.get("group", False) or (isinstance(sender, str) and "@g.us" in sender)
    participant = data.get("participant", "")
    identity    = participant if is_group and participant else sender

    print(f"[WEBHOOK] sender={sender}, participant={participant}, is_group={is_group}, identity={identity}")

    # ── Filter khusus grup: harus diawali trigger ──
    GROUP_TRIGGERS = ["!tanya", "/tanya", "!ai", "/ai", "bot:", "bot :"]
    if is_group:
        message_lower = message.lower()
        matched_trigger = None
        for trigger in GROUP_TRIGGERS:
            if message_lower.startswith(trigger):
                matched_trigger = trigger
                break

        if not matched_trigger:
            print(f"[GRUP] Pesan diabaikan (tidak ada trigger): {message[:50]}")
            return jsonify({"status": "ignored_no_trigger"}), 200

        message = message[len(matched_trigger):].strip()
        if not message:
            threading.Thread(
                target=send_wa,
                args=(sender, "❓ Pertanyaanmu kosong. Contoh: *!tanya berapa ICU critical di RU II?*"),
                daemon=True
            ).start()
            return jsonify({"status": "ok"}), 200

    # ── Cek apakah pengirim diizinkan ──
    if not is_group and identity not in ALLOWED_NUMBERS:
        print(f"Akses ditolak: {identity}")
        return jsonify({"status": "ignored"}), 200

    if not message:
        return jsonify({"status": "empty"}), 200

    # Command reset history
    if message.lower() in ["/reset", "reset", ".reset"]:
        clear_history(identity)
        threading.Thread(
            target=send_wa,
            args=(sender, "🔄 *Percakapan direset.* Memori sesi sebelumnya dihapus."),
            daemon=True
        ).start()
        return jsonify({"status": "ok"}), 200

    # Proses di background thread — reply 200 OK langsung ke Fonnte
    # agar Fonnte tidak timeout dan tidak kirim ulang webhook yang sama
    def process():
        try:
            answer = run_wa(message, identity)
            send_wa(sender, answer)
        except Exception as e:
            print(f"[ERROR] Gagal proses pesan dari {identity}: {e}")

    threading.Thread(target=process, daemon=True).start()
    return jsonify({"status": "ok"}), 200

@app.route("/sync/graph", methods=["GET"])
def sync_graph_endpoint():
    """Trigger sync manual via browser atau Postman."""
    key = request.args.get("key", "")
    if key != CHATBOT_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    threading.Thread(target=run_graph_sync, daemon=True).start()
    return jsonify({"status": "Sync graph dimulai di background!"}), 200


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "WA Bot is running 🚀"}), 200

# ─── 10. RUN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 WA Bot berjalan di port {port}...")
    app.run(host="0.0.0.0", port=port)