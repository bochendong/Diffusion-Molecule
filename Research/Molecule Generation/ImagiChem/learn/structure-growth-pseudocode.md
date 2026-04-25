# Structure growth — pseudocode (ImagiChem)

English pseudocode matching `assemble_from_input_string` in `ImagiChem code/imagichem_core.py`.  
Random choices use the global `RNG` / `random` after `set_seed` from the image.

---

## Intuitive pseudocode (shopping list + prefabs — no chemistry words)

Think: **`pool`** = how many tokens of each label (C, N, O, …) are left to spend. **`g`** = graph: **nodes** (tokens placed) and **edges** (connections). **Open connector** = a node that can still accept another edge.

### Phase A — Put down a big prefab or one lone token

```
procedure START_GRAPH(pool, g):
    repeat 1 or 2 times (2 only if total tokens in pool ≥ 12):
        pick a BIG_PREFAB from CORE_LIBRARY whose token cost ≤ pool (uniform among valid)
        if none: continue
        pay its cost from pool
        if g is empty: copy entire prefab into g; remember which nodes belong to first prefab
        else: attach second prefab to an open connector outside the first prefab,
              or pay one extra "C" token, insert a linker node on the first prefab, attach there
    if g is still empty:
        take the most common label still in pool, pay one token, add a single node to g
```

### Phase B — Spend the rest of the list: kits first, then one token at a time

```
while pool still has tokens:
    K ← in GROUP_LIBRARY whose cost ≤ pool and random draw passes their weight
    if K is not empty:
        pick one kit from K; find a node in g with an open connector (free slots first)
        if none: convert some double-edges to single-edges until some node opens; retry host
        if host found: pay kit cost; copy kit into g and glue one of its pins to host
        continue
    pop next token label from pool
    if nothing to pop: stop
    add one new node; glue it with a single edge to any open connector; if stuck, free slots then retry
```

### Phase C — If the graph is several islands, bridge them

```
procedure MERGE_ISLANDS(g):
    while more than one connected cluster and attempt budget not exceeded:
        take two smallest clusters; find any open connector in each; add one edge between them
        if stuck: try freeing slots by demoting double-edges; if still stuck, give up bridging
```

### Phase D — Export and tidy

```
procedure EXPORT(g):
    send g to external checker; if rejected, simplify edges and resend until accepted
    assign 2D positions for drawing; return final object
```

The detailed procedures below are the **same algorithm** with implementation names (`merge_graph_into`, `SanitizeMol`, …).

---

## Phase A — Initialize graph from cores (scaffold seeding)

```
procedure PLACE_CORES(pool, g):
    n_cores ← 1
    if sum(pool) ≥ 12 then
        n_cores ← random_choice({1, 2})

    atoms_in_first_core ← ∅

    for i ← 0 to n_cores − 1 do
        (core_name, core_factory) ← CHOOSE_CORE_BY_POOL(pool)
        if core_factory is NULL then
            continue

        needed_atoms ← requirements of that core in CORE_LIBRARY
        core_graph ← core_factory()

        host_for_second_core ← NULL
        if g is not empty AND i > 0 then
            possible_hosts ← { idx in g | idx ∉ atoms_in_first_core AND valence_rem[idx] ≥ 1 }
            if possible_hosts is not empty then
                host_for_second_core ← random_choice(possible_hosts)
            else if pool["C"] > 0 then
                needed_atoms["C"] ← needed_atoms["C"] + 1   // pay for a linker C
            else
                continue

        if pool does not contain needed_atoms then
            continue

        CONSUME_FROM_POOL(pool, needed_atoms)

        if g is empty then
            mapping ← MERGE_GRAPH_INTO(g, core_graph)   // no extra attachment bond
            atoms_in_first_core ← set of new indices in mapping
        else
            attach_src ← random_choice(keys(core_graph.elements))
            if host_for_second_core is not NULL then
                MERGE_GRAPH_INTO(g, core_graph,
                    attach_to_target = host_for_second_core,
                    attach_from_source_idx = attach_src)
            else
                linker_point ← random_choice(atoms_in_first_core) or random_choice(g.elements)
                if valence_rem[linker_point] ≥ 1 then
                    linker_idx ← NEW_ATOM(g, "C")
                    ADD_BOND(g, linker_point, linker_idx, 1)
                    MERGE_GRAPH_INTO(g, core_graph,
                        attach_to_target = linker_idx,
                        attach_from_source_idx = attach_src)

    if g is still empty then
        if pool is empty then ERROR
        elem ← most_common_element(pool)
        pool[elem] ← pool[elem] − 1
        NEW_ATOM(g, elem)
```

---

## Phase B — Iterative growth (groups then single atoms)

```
procedure GROW_UNTIL_POOL_EMPTY(pool, g):
    while sum(pool.values) > 0 do
        // 1) Try weighted functional-group / synthon fragments
        possible_groups ← []
        for each group_info in GROUP_LIBRARY do
            if pool satisfies group_info.requirements AND RNG.random() < group_info.weight then
                append group_info to possible_groups

        group_added ← false
        if possible_groups is not empty then
            chosen ← random_choice(possible_groups)
            host ← CHOOSE_HOST(g)
            if host is NULL then
                DEMOTE_SOME_DOUBLE_BONDS_UNTIL_CAPACITY(g, 1)
                host ← CHOOSE_HOST(g)

            if host is not NULL then
                CONSUME_FROM_POOL(pool, chosen.requirements)
                fragment ← chosen.factory()
                MERGE_GRAPH_INTO(g, fragment,
                    attach_to_target = host,
                    attach_from_source_idx = 0)
                group_added ← true

        if group_added then
            continue   // next while iteration

        // 2) Fallback: one atom from pool, connect by a single bond
        elem ← CHOOSE_NEXT_ELEM_FOR_CHAIN(pool)   // also decrements pool for elem
        if elem is NULL then
            break

        new_idx ← NEW_ATOM(g, elem)
        connected ← false
        hosts ← shuffled list of { idx | valence_rem[idx] ≥ 1 AND idx ≠ new_idx }

        for host in hosts do
            if ADD_BOND(g, host, new_idx, 1) succeeds then
                connected ← true
                break

        if not connected then
            DEMOTE_SOME_DOUBLE_BONDS_UNTIL_CAPACITY(g, 2)
            hosts ← { idx | valence_rem[idx] ≥ 1 AND idx ≠ new_idx }
            for host in hosts do
                if ADD_BOND(g, host, new_idx, 1) succeeds then
                    connected ← true
                    break
```

### Phase B — compact (≤10 lines)

```
while (atoms left in pool):
    C ← { entry ∈ GROUP_LIBRARY | pool ≥ entry.requirements and RNG < entry.weight }
    if C ≠ ∅:
        merge RNG.choice(C) onto CHOOSE_HOST(g), demoting doubles first if no host has free valence
        CONSUME_FROM_POOL for that entry; 
        continue
    e ← CHOOSE_NEXT_ELEM_FOR_CHAIN(pool)
    if e is None: 
        break
    NEW_ATOM(g, e); try ADD_BOND to any node with spare valence; if all fail, DEMOTE_SOME_DOUBLE_BONDS then retry bonds
```

---

## Phase C — Bridge disconnected components (optional repair)

Not “growth” in the chemical sense, but completes connectivity before RDKit export.

```
procedure BRIDGE_COMPONENTS(g):
    comps ← CONNECTED_COMPONENTS(g)
    attempts ← 0
    while |comps| > 1 AND attempts < 200 do
        sort comps by size ascending
        (A, B) ← (smallest, second smallest)

        node_a ← first node in A with valence_rem ≥ 1
        if node_a is NULL then
            DEMOTE_SOME_DOUBLE_BONDS_UNTIL_CAPACITY(g, total_capacity(g) + 1)
            node_a ← first node in A with valence_rem ≥ 1

        node_b ← first node in B with valence_rem ≥ 1
        if node_b is NULL then
            DEMOTE_SOME_DOUBLE_BONDS_UNTIL_CAPACITY(g, total_capacity(g) + 1)
            node_b ← first node in B with valence_rem ≥ 1

        if node_a is NULL or node_b is NULL then
            break   // cannot connect

        if ADD_BOND(g, node_a, node_b, 1) succeeds then
            comps ← CONNECTED_COMPONENTS(g)

        attempts ← attempts + 1
```

---

## Phase D — Export and sanitize (summary)

```
procedure EXPORT_MOL(g):
    mol ← GRAPH_TO_RWMOL(g)
    try SANITIZE(mol)
    catch
        repeatedly DEMOTE_BONDS on graph until sanitize succeeds
    COMPUTE_2D_COORDS(mol)
    return mol
```

---

## Helpers (abbreviated semantics)

| Name | Role |
|------|------|
| `CHOOSE_CORE_BY_POOL` | Among cores whose `requirements` ⊆ `pool`, pick one uniformly. |
| `CHOOSE_HOST` | Prefer nodes with spare valence; biased toward lower local degree / higher remaining valence. |
| `MERGE_GRAPH_INTO` | Copy all atoms and bonds from fragment into `g`; optionally add one single bond from a host atom to `attach_from_source_idx` in the fragment. |
| `DEMOTE_SOME_DOUBLE_BONDS_UNTIL_CAPACITY` | Turn some double bonds into single bonds until enough nodes have free valence (heuristic order: prefer non‑C–C doubles last). |

---

## Related source

- `ImagiChem code/imagichem_core.py` — `assemble_from_input_string`, `merge_graph_into`, `choose_core_by_pool`, `choose_host`, …
- `ImagiChem code/groups.py` — `GROUP_LIBRARY`
- `ImagiChem code/cores.py` — `CORE_LIBRARY`
