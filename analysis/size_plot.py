# import matplotlib as plt

# fig, axes = plt.subplots(2, 3, figsize=(13, 8))
# ax1, ax2, ax3 = axes[0]
# ax4, ax5, ax6 = axes[1]

# for ax in axes.flat:
#     ax.set_facecolor("#EEF2F6")

# colors = cm.Dark2(range(7))

# line1 = ax1.plot(
#     sizes, neg136simp_acc, label="NEG-136-SIMP", color=colors[0], marker="o"
# )
# line2 = ax1.plot(
#     sizes, neg1500temp_acc, label="NEG-1500-SIMP-TEMP", color=colors[1], marker="o"
# )
# line3 = ax1.plot(
#     sizes, neg1500gen_acc, label="NEG-1500-SIMP-GEN", color=colors[2], marker="o"
# )
# ax1.set_title("Top-5 Accuracy")
# ax1.set_xlim(sizes[0], sizes[-1])
# ax1.set_ylim(-0.05, 1.05)
# ax1.yaxis.set_major_locator(YTICK_LOCATOR)
# ax1.grid(True, linestyle="--", alpha=0.7, color="#CCCCCC")
# ax1.tick_params(axis="x", rotation=45)

# ax2.plot(sizes, neg136simp_sh, color=colors[0], marker="o")
# ax2.plot(sizes, neg1500temp_sh, color=colors[1], marker="o")
# ax2.plot(sizes, neg1500gen_sh, color=colors[2], marker="o")
# ax2.set_title("S*")
# ax2.set_xlim(sizes[0], sizes[-1])
# ax2.set_ylim(-0.05, 1.05)
# ax2.yaxis.set_major_locator(YTICK_LOCATOR)
# ax2.grid(True, linestyle="--", alpha=0.7, color="#CCCCCC")
# ax2.tick_params(axis="x", rotation=45)

# line4 = ax3.plot(
#     sizes, neg136simp_aff, label="NEG-136-SIMP (Aff)", color=colors[0], marker="o"
# )
# line5 = ax3.plot(
#     sizes,
#     neg1500temp_aff,
#     label="NEG-1500-SIMP-TEMP (Aff)",
#     color=colors[1],
#     marker="o",
# )
# line6 = ax3.plot(
#     sizes, neg1500gen_aff, label="NEG-1500-SIMP-GEN (Aff)", color=colors[2], marker="o"
# )
# line7 = ax3.plot(
#     sizes,
#     neg136simp_neg,
#     label="NEG-136-SIMP (Neg)",
#     color=colors[0],
#     marker="o",
#     linestyle="--",
# )
# line8 = ax3.plot(
#     sizes,
#     neg1500temp_neg,
#     label="NEG-1500-SIMP-TEMP (Neg)",
#     color=colors[1],
#     marker="o",
#     linestyle="--",
# )
# line9 = ax3.plot(
#     sizes,
#     neg1500gen_neg,
#     label="NEG-1500-SIMP-GEN (Neg)",
#     color=colors[2],
#     marker="o",
#     linestyle="--",
# )
# ax3.set_title("S(+/-)")
# ax3.set_xlim(sizes[0], sizes[-1])
# ax3.set_ylim(-0.05, 1.05)
# ax3.yaxis.set_major_locator(YTICK_LOCATOR)
# ax3.grid(True, linestyle="--", alpha=0.7, color="#CCCCCC")
# ax3.tick_params(axis="x", rotation=45)

# handles1 = [line1[0], line2[0], line3[0]]
# labels1 = [
#     r"$NEG\text{-}136\text{-}SIMP$",
#     r"$NEG\text{-}1500\text{-}SIMP_{TEMP}$",
#     r"$NEG\text{-}1500\text{-}SIMP_{GEN}$",
# ]
# solid_line = line4[0]
# dashed_line = line7[0]
# handles3 = [solid_line, dashed_line]
# labels3 = ["Affirmative", "Negative"]

# fig.legend(
#     handles1 + handles3,
#     labels1 + labels3,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 0.95),
#     ncol=5,
#     frameon=True,
# )

# line10 = ax4.plot(
#     sizes, neg136natln_acc, label="NEG-136-NAT-LN", color=colors[4], marker="o"
# )
# line11 = ax4.plot(
#     sizes, neg136natnt_acc, label="NEG-136-NAT-NT", color=colors[5], marker="o"
# )
# ax4.set_xlim(sizes[0], sizes[-1])
# ax4.set_ylim(-0.05, 1.05)
# ax4.yaxis.set_major_locator(YTICK_LOCATOR)
# ax4.grid(True, linestyle="--", alpha=0.7, color="#CCCCCC")
# ax4.tick_params(axis="x", rotation=45)

# ax5.plot(sizes, neg136natln_sh, color=colors[4], marker="o")
# ax5.plot(sizes, neg136natnt_sh, color=colors[5], marker="o")
# ax5.set_xlim(sizes[0], sizes[-1])
# ax5.set_ylim(-0.05, 1.05)
# ax5.yaxis.set_major_locator(YTICK_LOCATOR)
# ax5.grid(True, linestyle="--", alpha=0.7, color="#CCCCCC")
# ax5.tick_params(axis="x", rotation=45)

# line12 = ax6.plot(
#     sizes, neg136natln_aff, label="NEG-136-NAT-LN (Aff)", color=colors[4], marker="o"
# )
# line13 = ax6.plot(
#     sizes, neg136natnt_aff, label="NEG-136-NAT-NT (Aff)", color=colors[5], marker="o"
# )
# line14 = ax6.plot(
#     sizes,
#     neg136natln_neg,
#     label="NEG-136-NAT-LN (Neg)",
#     color=colors[4],
#     marker="o",
#     linestyle="--",
# )
# line15 = ax6.plot(
#     sizes,
#     neg136natnt_neg,
#     label="NEG-136-NAT-NT (Neg)",
#     color=colors[5],
#     marker="o",
#     linestyle="--",
# )
# ax6.set_xlim(sizes[0], sizes[-1])
# ax6.set_ylim(-0.05, 1.05)
# ax6.yaxis.set_major_locator(YTICK_LOCATOR)
# ax6.grid(True, linestyle="--", alpha=0.7, color="#CCCCCC")
# ax6.tick_params(axis="x", rotation=45)

# handles4 = [line12[0], line14[0]]
# labels4 = ["Affirmative", "Negative"]
# handles_bottom = [line10[0], line11[0]]
# labels_bottom = [
#     r"$NEG\text{-}136\text{-}NAT_{LN}$",
#     r"$NEG\text{-}136\text{-}NAT_{NT}$",
# ]

# fig.legend(
#     handles_bottom + handles4,
#     labels_bottom + labels4,
#     loc="lower center",
#     bbox_to_anchor=(0.5, 0.07),
#     ncol=4,
#     frameon=True,
# )
# plt.tight_layout()
# plt.subplots_adjust(top=0.85, bottom=0.2, hspace=0.24)

# plt.savefig("pythia_size.pdf", dpi=300, bbox_inches="tight")
