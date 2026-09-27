"""
Mass Spectrum Visualization Engine
Generates publication-quality stick spectra and multi-energy comparison plots using Matplotlib.
"""

from typing import List, Optional, Tuple
import os

import matplotlib
# Use non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.spectrum import MassSpectrum


class SpectrumVisualizer:
    """Renders single stick spectra and multi-energy comparative plots."""

    def __init__(self, style: str = "white"):
        self.style = style
        if style == "dark":
            plt.style.use('dark_background')
            self.line_color = "#00d2ff"
            self.base_color = "#ff4757"
            self.mol_ion_color = "#2ed573"
            self.text_color = "#ffffff"
        else:
            plt.style.use('default')
            self.line_color = "#1e3799"
            self.base_color = "#eb2f06"
            self.mol_ion_color = "#009432"
            self.text_color = "#2f3542"

    def plot_spectrum(
        self,
        spectrum: MassSpectrum,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        top_n_labels: int = 8
    ) -> plt.Figure:
        """
        Plot a stick mass spectrum with annotated peaks.
        """
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        mzs = [p.mz for p in spectrum.sorted_peaks]
        intensities = [p.intensity for p in spectrum.sorted_peaks]

        if not mzs:
            ax.text(0.5, 0.5, "No peaks predicted", ha="center", va="center")
            return fig

        max_mz = max(mzs) + 15
        min_mz = max(0, min(mzs) - 10)

        # Draw baseline and vertical sticks
        ax.axhline(0, color=self.text_color, linewidth=0.8, alpha=0.5)

        for peak in spectrum.sorted_peaks:
            color = self.line_color
            linewidth = 1.8

            if peak.is_base_peak:
                color = self.base_color
                linewidth = 2.4
            elif peak.is_molecular_ion:
                color = self.mol_ion_color
                linewidth = 2.2

            ax.vlines(peak.mz, 0, peak.intensity, color=color, linewidth=linewidth, alpha=0.9)

        # Label top peaks
        top_peaks = spectrum.get_top_peaks(top_n_labels)
        labeled_mzs = set()

        for peak in top_peaks:
            labeled_mzs.add(peak.mz)
            label = f"{peak.mz}"
            if peak.is_molecular_ion:
                label += " (M+•)"
            elif peak.primary_formula:
                label += f"\n[{peak.primary_formula}]+"

            ax.annotate(
                label,
                xy=(peak.mz, peak.intensity),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if peak.is_base_peak else "normal",
                color=self.base_color if peak.is_base_peak else self.text_color
            )

        # Also label molecular ion if not in top peaks
        if spectrum.molecular_ion_mz in spectrum.peaks and spectrum.molecular_ion_mz not in labeled_mzs:
            m_peak = spectrum.peaks[spectrum.molecular_ion_mz]
            ax.annotate(
                f"{m_peak.mz} (M+•)",
                xy=(m_peak.mz, m_peak.intensity),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=self.mol_ion_color,
                fontweight="bold"
            )

        # Formatting
        chart_title = title or (
            f"Predicted EI Mass Spectrum: {spectrum.molecule.formula} ({spectrum.molecule.smiles})\n"
            f"Electron Energy: {spectrum.beam_energy_ev} eV | Base Peak m/z: {spectrum.base_peak_mz}"
        )
        ax.set_title(chart_title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("m/z", fontsize=10, fontweight="bold")
        ax.set_ylabel("Relative Intensity (%)", fontsize=10, fontweight="bold")
        ax.set_xlim(min_mz, max_mz)
        ax.set_ylim(0, 115)

        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_energy_comparison(
        self,
        spectra: List[MassSpectrum],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot stacked comparison of mass spectra across multiple beam energies (e.g. 20, 40, 70, 100 eV).
        """
        if not spectra:
            raise ValueError("No spectra provided for comparison.")

        num_spectra = len(spectra)
        mol = spectra[0].molecule
        all_mzs = [p.mz for s in spectra for p in s.sorted_peaks]
        max_mz = (max(all_mzs) + 10) if all_mzs else 150
        min_mz = max(0, min(all_mzs) - 10) if all_mzs else 0

        fig, axes = plt.subplots(num_spectra, 1, figsize=(11, 2.5 * num_spectra), sharex=True, dpi=300)
        if num_spectra == 1:
            axes = [axes]

        fig.suptitle(
            f"Electron Beam Energy Dependence Study: {mol.formula} ({mol.smiles})\n"
            "Comparison across 20 eV, 40 eV, 70 eV, and 100 eV",
            fontsize=13, fontweight="bold", y=0.99
        )

        for i, (ax, spec) in enumerate(zip(axes, spectra)):
            # Draw sticks
            ax.axhline(0, color="#747d8c", linewidth=0.8)
            for peak in spec.sorted_peaks:
                col = self.base_color if peak.is_base_peak else (self.mol_ion_color if peak.is_molecular_ion else self.line_color)
                ax.vlines(peak.mz, 0, peak.intensity, color=col, linewidth=1.7)

            # Annotate top 5 peaks
            for p in spec.get_top_peaks(5):
                ax.text(p.mz, p.intensity + 3, f"{p.mz}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

            # Mark Molecular Ion
            if spec.molecular_ion_mz in spec.peaks:
                m_ion = spec.peaks[spec.molecular_ion_mz]
                ax.text(m_ion.mz, m_ion.intensity + 3, f"{m_ion.mz}\n(M+•)", ha="center", va="bottom", fontsize=7, color=self.mol_ion_color)

            ax.set_ylabel(f"{spec.beam_energy_ev:.0f} eV\nRel. Int (%)", fontsize=9, fontweight="bold")
            ax.set_ylim(0, 120)
            ax.grid(axis='y', linestyle=':', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[-1].set_xlabel("m/z", fontsize=10, fontweight="bold")
        axes[-1].set_xlim(min_mz, max_mz)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
