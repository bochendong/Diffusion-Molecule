Building on the controlled composition we just covered, this slide shows how ImagiChem turns that atomic pool into a valid molecule through hierarchical assembly. The key shift here is that generation is not a random join operation; it is a staged construction process.

At a high level, the pipeline moves from an atomic pool to a scaffold core, then to synthon growth, and finally to a valence-consistent structure. That ordering matters because the hierarchy enforces chemical feasibility early, instead of trying to repair a chaotic structure at the end.

Step one is core selection. ImagiChem begins from curated fragments and drug-like scaffolds, not from arbitrary atom strings. A candidate core must also fit the available atomic pool, so the model cannot choose a scaffold that demands atoms or functionalities that were never implied by the image-derived composition.

Step two is structure growth. Once a compatible core is chosen, synthons and functional groups are added iteratively. The attachment process follows weighted rules and checks available valence, which means growth is guided by both preference and constraint. In implementation terms, you can think of this as expanding a partial graph while continuously verifying legal attachment sites.

Step three is path recovery. If the current growth path reaches a dead end, the system does not simply fail. It can use fallback atom insertion or bond relaxation to reopen feasible construction routes. This is important because it preserves progress while avoiding brittle behavior when local choices temporarily block completion.

The result is a deliberate bias toward chemically meaningful assembly. Instead of random concatenation, the hierarchy prefers coherent substructures, controllable expansion, and structures that remain consistent with chemical rules. That is the central advantage of this stage: validity is built into the construction sequence itself.

So the main takeaway from this page is simple: ImagiChem first chooses a compatible core, then grows the molecule with valence-aware rules, and finally uses recovery strategies to avoid dead ends. This hierarchy makes the output far more reliable and chemically coherent, which sets us up naturally for the next topic: explicit chemical validity checks, filtering, and ranking.
