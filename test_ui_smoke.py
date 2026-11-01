"""UI smoke tests for QuantPulse Streamlit application.

Verifies that the application loads and each of the six navigation tabs
renders completely without raising runtime exceptions.
"""

from __future__ import annotations

import unittest
from streamlit.testing.v1 import AppTest


class QuantPulseUISmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.at = AppTest.from_file("app.py", default_timeout=30)
        cls.at.run()
        if cls.at.exception:
            raise RuntimeError(f"Initial load failed with exception: {cls.at.exception[0]}")

    def test_initial_load_succeeds(self):
        self.assertEqual(len(self.at.exception), 0)

    def test_market_overview_olap_tab(self):
        self.at.sidebar.radio[0].set_value("🌐 Market Overview & OLAP").run()
        self.assertEqual(len(self.at.exception), 0)

    def test_exploratory_analytics_tab(self):
        self.at.sidebar.radio[0].set_value("📊 Exploratory Analytics").run()
        self.assertEqual(len(self.at.exception), 0)

    def test_predictive_modeling_tab(self):
        self.at.sidebar.radio[0].set_value("⚡ Predictive Modeling").run()
        self.assertEqual(len(self.at.exception), 0)

    def test_asset_clusters_risk_tab(self):
        self.at.sidebar.radio[0].set_value("🎯 Asset Clusters & Risk").run()
        self.assertEqual(len(self.at.exception), 0)

    def test_co_movement_engine_tab(self):
        self.at.sidebar.radio[0].set_value("🔄 Co-Movement Engine").run()
        self.assertEqual(len(self.at.exception), 0)

    def test_technical_architecture_tab(self):
        self.at.sidebar.radio[0].set_value("📄 Technical Architecture").run()
        self.assertEqual(len(self.at.exception), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
