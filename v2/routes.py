"""Registrasi endpoint v2 (§3 dokumen M3) ke sebuah Flask app yang sudah ada.

Dipanggil dari app.py dan app_public.py lewat `register_v2_routes(app)` supaya
v1 lama tetap jalan di proses/host yang sama (§3.4 — klien jatuh ke v1 hanya
saat v2 membalas 404/405, jadi keduanya harus hidup berdampingan).
"""
from __future__ import annotations

import os
import time

from flask import jsonify, request

from . import forecast_service

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_CSV = os.path.join(BASE_DIR, "Models", "backtest", "backtest_summary.csv")
BACKTEST_DECISION = os.path.join(BASE_DIR, "Dokumen", "M3 - Hasil Backtest.md")

_startup_time = time.time()


def register_v2_routes(app):
    @app.route("/api/health", methods=["GET"])
    def v2_health():
        from . import lstm_adapter
        model_loaded = lstm_adapter.is_ready()
        # "Warm" setelah proses berjalan cukup lama DAN model sudah dimuat sekali —
        # cold start HF Space yang sesungguhnya diukur oleh klien lewat endpoint ini (§3.1).
        warm = model_loaded and (time.time() - _startup_time) > 5
        return jsonify({
            "status": "ok",
            "model_loaded": model_loaded,
            "model_version": forecast_service.MODEL_VERSION,
            "warm": warm,
        }), 200

    @app.route("/api/v2/forecast", methods=["POST"])
    def v2_forecast():
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Body JSON kosong atau tidak valid."}), 400
        try:
            response = forecast_service.build_forecast_response(payload)
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"Gagal membangun forecast: {e}"}), 500
        return jsonify(response), 200

    @app.route("/api/v2/backtest", methods=["GET"])
    def v2_backtest():
        """Laporan backtest §6 — dipakai untuk lampiran skripsi, bukan oleh app."""
        if not os.path.exists(BACKTEST_CSV):
            return jsonify({
                "error": "Backtest belum dijalankan.",
                "hint": "Jalankan: python evaluation/backtest.py",
            }), 404

        import pandas as pd
        df = pd.read_csv(BACKTEST_CSV)

        as_csv = request.args.get("format") == "csv"
        if as_csv:
            return app.response_class(df.to_csv(index=False), mimetype="text/csv")

        decision = None
        if os.path.exists(BACKTEST_DECISION):
            with open(BACKTEST_DECISION, "r", encoding="utf-8") as f:
                decision = f.read()

        return jsonify({
            "results": df.to_dict(orient="records"),
            "decision_document": decision,
        }), 200

    return app
