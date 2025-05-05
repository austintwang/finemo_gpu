import os
import importlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
from jinja2 import Template

from . import templates


def abbreviate_motif_name(name):
    try:
        group, motif = name.split(".")
        if group == "pos_patterns":
            group_short = "+"
        elif group == "neg_patterns":
            group_short = "-"
        else:
            raise Exception
        motif_num = motif.split("_")[1]
        return f"{group_short}/{motif_num}"
    except:
        return name


def plot_hit_stat_distributions(hits_df, motif_names, plot_dir):
    hits_df = hits_df.collect()
    hits_by_motif = hits_df.partition_by("motif_name", as_dict=True)
    dummy_df = hits_df.clear()

    motifs_dir = os.path.join(plot_dir, "motif_stat_distributions")
    os.makedirs(motifs_dir, exist_ok=True)
    for m in motif_names:
        hits = hits_by_motif.get((m,), dummy_df)
        coefficients = hits.get_column("hit_coefficient_global").to_numpy()
        similarities = hits.get_column("hit_similarity").to_numpy()
        importances = hits.get_column("hit_importance").to_numpy()

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.hist(coefficients, bins=50, density=True)

        output_path_png = os.path.join(motifs_dir, f"{m}_coefficients.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}_coefficients.svg")
        plt.savefig(output_path_svg)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.hist(similarities, bins=50, density=True)

        output_path_png = os.path.join(motifs_dir, f"{m}_similarities.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}_similarities.svg")
        plt.savefig(output_path_svg)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.hist(importances, bins=50, density=True)

        output_path_png = os.path.join(motifs_dir, f"{m}_importances.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}_importances.svg")
        plt.savefig(output_path_svg)
        plt.close(fig)


def plot_hit_peak_distributions(occ_df, motif_names, plot_dir):
    motifs_dir = os.path.join(plot_dir, "motif_hit_distributions")
    os.makedirs(motifs_dir, exist_ok=True)

    for m in motif_names:
        fig, ax = plt.subplots(figsize=(5, 2))

        unique, counts = np.unique(occ_df.get_column(m), return_counts=True)
        freq = counts / counts.sum()
        num_bins = np.amax(unique, initial=0) + 1
        x = np.arange(num_bins)
        y = np.zeros(num_bins)
        y[unique] = freq
        ax.bar(x, y)

        output_path_png = os.path.join(motifs_dir, f"{m}.png")
        plt.savefig(output_path_png, dpi=300)
        output_path_svg = os.path.join(motifs_dir, f"{m}.svg")
        plt.savefig(output_path_svg)

        plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 4))

    unique, counts = np.unique(occ_df.get_column("total"), return_counts=True)
    freq = counts / counts.sum()
    num_bins = np.amax(unique, initial=0) + 1
    x = np.arange(num_bins)
    y = np.zeros(num_bins)
    y[unique] = freq
    ax.bar(x, y)

    ax.set_xlabel("Total hits per region")
    ax.set_ylabel("Frequency")

    output_path_png = os.path.join(plot_dir, "total_hit_distribution.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(plot_dir, "total_hit_distribution.svg")
    plt.savefig(output_path_svg, dpi=300)

    plt.close(fig)


def plot_peak_motif_indicator_heatmap(peak_hit_counts, motif_names, output_dir):
    """
    Plots a simple indicator heatmap of the motifs in each peak.
    """
    cov_norm = 1 / np.sqrt(np.diag(peak_hit_counts))
    matrix = peak_hit_counts * cov_norm[:, None] * cov_norm[None, :]
    motif_keys = [abbreviate_motif_name(m) for m in motif_names]

    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    
    # Plot the heatmap
    cax = ax.imshow(matrix, interpolation="nearest", aspect="equal", cmap="Greens")

    # Set axes on heatmap
    ax.set_yticks(np.arange(len(motif_keys)))
    ax.set_yticklabels(motif_keys)
    ax.set_xticks(np.arange(len(motif_keys)))
    ax.set_xticklabels(motif_keys, rotation=90)
    ax.set_xlabel("Motif i")
    ax.set_ylabel("Motif j")

    ax.tick_params(axis='both', labelsize=8)

    cbar = fig.colorbar(cax, ax=ax, orientation="vertical", shrink=0.6, aspect=30)
    cbar.ax.tick_params(labelsize=8) 
    
    output_path_png = os.path.join(output_dir, "motif_cooocurrence.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(output_dir, "motif_cooocurrence.svg")
    plt.savefig(output_path_svg)

    plt.close()


def plot_seqlet_confusion_heatmap(seqlet_confusion, motif_names, output_dir):
    motif_keys = [abbreviate_motif_name(m) for m in motif_names]

    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    
    # Plot the heatmap
    cax = ax.imshow(seqlet_confusion, interpolation="nearest", aspect="equal", cmap="Blues")

    # Set axes on heatmap
    ax.set_yticks(np.arange(len(motif_keys)))
    ax.set_yticklabels(motif_keys)
    ax.set_xticks(np.arange(len(motif_keys)))
    ax.set_xticklabels(motif_keys, rotation=90)
    ax.set_xlabel("Hit motif")
    ax.set_ylabel("Seqlet motif")

    ax.tick_params(axis='both', labelsize=8)

    cbar = fig.colorbar(cax, ax=ax, orientation="vertical", shrink=0.6, aspect=30)
    cbar.ax.tick_params(labelsize=8) 

    output_path_png = os.path.join(output_dir, "seqlet_confusion.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(output_dir, "seqlet_confusion.svg")
    plt.savefig(output_path_svg)

    plt.close()


class LogoGlyph(AbstractPathEffect):
    def __init__(self, glyph, ref_glyph='E', font_props=None,
                 offset=(0., 0.), **kwargs):

        super().__init__(offset)

        path_orig = TextPath((0, 0), glyph, size=1, prop=font_props)
        dims = path_orig.get_extents()
        ref_dims = TextPath((0, 0), ref_glyph, size=1, prop=font_props).get_extents()

        h_scale = 1 / dims.height
        ref_width = max(dims.width, ref_dims.width)
        w_scale = 1 / ref_width
        w_shift = (1 - dims.width / ref_width) / 2
        x_shift = -dims.x0
        y_shift = -dims.y0
        stretch = (
            Affine2D()
            .translate(tx=x_shift, ty=y_shift)
            .scale(sx=w_scale, sy=h_scale)
            .translate(tx=w_shift, ty=0)
        )

        self.path = stretch.transform_path(path_orig)

        #: The dictionary of keywords to update the graphics collection with.
        self._gc = kwargs

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        return renderer.draw_path(gc, self.path, affine, rgbFace)


def plot_logo(ax, heights, glyphs, colors=None, font_props=None, shade_bounds=None):
    if colors is None:
        colors = {g: None for g in glyphs}

    ax.margins(x=0, y=0)
    
    pos_values = np.clip(heights, 0, None)
    neg_values = np.clip(heights, None, 0)
    pos_order = np.argsort(pos_values, axis=0)
    neg_order = np.argsort(neg_values, axis=0)[::-1, :]
    pos_reorder = np.argsort(pos_order, axis=0)
    neg_reorder = np.argsort(neg_order, axis=0)
    pos_offsets = np.take_along_axis(
        np.cumsum(
            np.take_along_axis(pos_values, pos_order, axis=0), axis=0
        ), pos_reorder, axis=0
    )
    neg_offsets = np.take_along_axis(
        np.cumsum(
            np.take_along_axis(neg_values, neg_order, axis=0), axis=0
        ), neg_reorder, axis=0
    )
    bottoms = pos_offsets + neg_offsets - heights

    x = np.arange(heights.shape[1])

    for glyph, height, bottom in zip(glyphs, heights, bottoms):
        ax.bar(x, height, 0.95, bottom=bottom, 
               path_effects=[LogoGlyph(glyph, font_props=font_props)], color=colors[glyph])

    if shade_bounds is not None:
        start, end = shade_bounds
        ax.axvspan(start - 0.5, end - 0.5, color='0.9', zorder=-1)

    ax.axhline(zorder=-1, linewidth=0.5, color='black')


LOGO_ALPHABET = 'ACGT'
LOGO_COLORS = {"A": '#109648', "C": '#255C99', "G": '#F7B32B', "T": '#D62839'}
LOGO_FONT = FontProperties(weight="bold")


def plot_cwms(cwms, trim_bounds, out_dir, alphabet=LOGO_ALPHABET, colors=LOGO_COLORS, font=LOGO_FONT):
    for m, v in cwms.items():
        motif_dir = os.path.join(out_dir, m)
        os.makedirs(motif_dir, exist_ok=True)
        for cwm_type, cwm in v.items():
            fig, ax = plt.subplots(figsize=(10, 2))

            plot_logo(ax, cwm, alphabet, colors=colors, font_props=font, shade_bounds=trim_bounds[m][cwm_type])

            for name, spine in ax.spines.items():
                spine.set_visible(False)
            
            output_path_png = os.path.join(motif_dir, f"{cwm_type}.png")
            plt.savefig(output_path_png, dpi=100)
            output_path_svg = os.path.join(motif_dir, f"{cwm_type}.svg")
            plt.savefig(output_path_svg)

            plt.close(fig)


def plot_hit_vs_seqlet_counts(recall_data, output_dir):
    x = []
    y = []
    m = []
    for k, v in recall_data.items():
        x.append(v["num_hits_total"])
        y.append(v["num_seqlets"])
        m.append(k)

    lim = max(np.amax(x), np.amax(y))

    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    ax.axline((0, 0), (lim, lim), color="0.3", linewidth=0.7, linestyle=(0, (5, 5)))
    ax.scatter(x, y, s=5)
    for i, txt in enumerate(m):
        short = abbreviate_motif_name(txt)
        ax.annotate(short, (x[i], y[i]), fontsize=8, weight="bold")

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel("Hits per motif")
    ax.set_ylabel("Seqlets per motif")

    output_path_png = os.path.join(output_dir, "hit_vs_seqlet_counts.png")
    plt.savefig(output_path_png, dpi=300)
    output_path_svg = os.path.join(output_dir, "hit_vs_seqlet_counts.svg")
    plt.savefig(output_path_svg)

    plt.close()


def write_report(report_df, motif_names, out_path, compute_recall, use_seqlets):
    template_str = importlib.resources.files(templates).joinpath('report.html').read_text()
    template = Template(template_str)
    report = template.render(report_data=report_df.iter_rows(named=True), 
                             motif_names=motif_names, compute_recall=compute_recall, 
                             use_seqlets=use_seqlets)
    with open(out_path, "w") as f:
        f.write(report)