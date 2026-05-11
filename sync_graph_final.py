import os
import psycopg2
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
NEO4J_URI    = "bolt://viaduct.proxy.rlwy.net:22569"
NEO4J_USER   = "neo4j"
NEO4J_PASS   = "Neo4j@2024"

# ─── KONEKSI ──────────────────────────────────────────────────────────────────
pg  = psycopg2.connect(DATABASE_URL)
neo = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ─── INDEX (jalankan sekali, biar query cepat) ────────────────────────────────
def create_indexes():
    with neo.session() as session:
        session.run("CREATE INDEX eq_id IF NOT EXISTS FOR (e:Equipment) ON (e.equipment)")
        session.run("CREATE INDEX boc_id IF NOT EXISTS FOR (b:BOC) ON (b.equipment)")
    print("✅ Index siap")

# ─── SYNC MASTER DATA EQUIPMENT ───────────────────────────────────────────────
def sync_equipment(batch_size=1000):
    cur = pg.cursor("eq_cursor")
    cur.execute("""
        SELECT
            equipment,
            criticality,
            functional_location,
            maintenance_plant,
            location,
            cost_center,
            equipment_category,
            description,
            manufacturer,
            model_type,
            technical_obj_type,
            material,
            material_description,
            size_dimension,
            sort_field_ata
        FROM master_data_equipment
        WHERE equipment IS NOT NULL
    """)

    total = 0
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break

        with neo.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (e:Equipment {equipment: row.equipment})
                SET
                    e.criticality          = row.criticality,
                    e.functional_location  = row.functional_location,
                    e.maintenance_plant    = row.maintenance_plant,
                    e.location             = row.location,
                    e.cost_center          = row.cost_center,
                    e.equipment_category   = row.equipment_category,
                    e.description          = row.description,
                    e.manufacturer         = row.manufacturer,
                    e.model_type           = row.model_type,
                    e.technical_obj_type   = row.technical_obj_type,
                    e.material             = row.material,
                    e.material_description = row.material_description,
                    e.size_dimension       = row.size_dimension,
                    e.sort_field_ata       = row.sort_field_ata
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

        total += len(rows)
        print(f"  Equipment: {total} node diproses...", end="\r")

    cur.close()
    print(f"\n✅ Equipment selesai: {total} node")

# ─── SYNC BOC (data performa/reliability equipment) ───────────────────────────
def sync_boc(batch_size=1000):
    cur = pg.cursor("boc_cursor")
    cur.execute("""
        SELECT
            equipment,
            ru,
            area,
            unit,
            grup_equipment,
            status,
            frequency,
            running_hours,
            mttr,
            mtbf,
            hasil
        FROM boc
        WHERE equipment IS NOT NULL
    """)

    total = 0
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break

        with neo.session() as session:
            session.run("""
                UNWIND $rows AS row

                MERGE (e:Equipment {equipment: row.equipment})

                MERGE (b:BOC {equipment: row.equipment})
                SET
                    b.ru            = row.ru,
                    b.area          = row.area,
                    b.unit          = row.unit,
                    b.grup_equipment= row.grup_equipment,
                    b.status        = row.status,
                    b.frequency     = row.frequency,
                    b.running_hours = row.running_hours,
                    b.mttr          = row.mttr,
                    b.mtbf          = row.mtbf,
                    b.hasil         = row.hasil

                MERGE (e)-[:PUNYA_PERFORMA]->(b)
            """, rows=[{
                "equipment":     r[0] or "",
                "ru":            r[1] or "",
                "area":          r[2] or "",
                "unit":          r[3] or "",
                "grup_equipment":r[4] or "",
                "status":        r[5] or "",
                "frequency":     r[6] or 0,
                "running_hours": float(r[7]) if r[7] else 0.0,
                "mttr":          float(r[8]) if r[8] else 0.0,
                "mtbf":          float(r[9]) if r[9] else 0.0,
                "hasil":         r[10] or ""
            } for r in rows])

        total += len(rows)
        print(f"  BOC: {total} relasi diproses...", end="\r")

    cur.close()
    print(f"\n✅ BOC selesai: {total} relasi")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔄 Mulai sync ke Neo4j...\n")

    create_indexes()
    sync_equipment()
    sync_boc()

    neo.close()
    pg.close()

    print("\n🎉 Knowledge graph berhasil dibangun!")
    print("\nContoh query yang bisa dipakai chatbot:")
    print("  Semua equipment di area tertentu:")
    print("    MATCH (e:Equipment)-[:PUNYA_PERFORMA]->(b:BOC) WHERE b.area = 'XXX' RETURN e, b")
    print("  Equipment dengan MTBF rendah:")
    print("    MATCH (e:Equipment)-[:PUNYA_PERFORMA]->(b:BOC) WHERE b.mtbf < 100 RETURN e, b ORDER BY b.mtbf")
    print("  Equipment critical yang sering breakdown:")
    print("    MATCH (e:Equipment)-[:PUNYA_PERFORMA]->(b:BOC) WHERE e.criticality = 'A' AND b.mttr > 10 RETURN e, b")