# Codebase Refactoring Recommendations

This document outlines recommendations for cleaning up the existing code into more manageable classes.

## Current State Summary

| Metric | Value |
|--------|-------|
| Total Python Lines | 781 |
| Number of Classes | 10 |
| Standalone Functions | 6 |
| Largest Class | `Protein` (222 lines) |
| Duplicate Code Files | 1 (`rotation_testing.py`) |
| Empty Stubs | 1 (`rot_flow.py`) |

---

## High Priority Refactoring

### 1. Split the `Protein` Class

**File:** `flow_testing/data/protein.py` (222 lines)

**Problem:** The `Protein` class is doing too much (God Class anti-pattern). It handles:
- Data representation (7 attributes)
- I/O operations (5 methods)
- Coordinate transformations (3 methods)

**Solution:** Split into focused classes:

| New Class | Responsibility | Methods to Move |
|-----------|----------------|-----------------|
| `ProteinIO` | File I/O operations | `from_pdb()`, `from_biotite()`, `to_pdb()`, `to_biotite()` |
| `Protein` | Core data structure | Keep attributes only (aa_type, atom_positions, etc.) |
| `ProteinFrameBuilder` | Frame construction | `to_bb_rigid()`, `to_psi_sin_cos()` |
| `ProteinTransform` | Coordinate transforms | `center()` and future transforms |

**Example refactored structure:**

```python
# protein.py - Data only
@dataclass
class Protein:
    aa_type: np.ndarray
    atom_positions: np.ndarray
    atom_names: np.ndarray
    atom_mask: np.ndarray
    chain_ids: np.ndarray
    residue_ids: np.ndarray
    b_factors: np.ndarray

# protein_io.py - I/O operations
class ProteinIO:
    @classmethod
    def from_pdb(cls, pdb_path: str) -> Protein: ...
    @classmethod
    def from_biotite(cls, atom_array: AtomArray) -> Protein: ...
    @staticmethod
    def to_pdb(protein: Protein, pdb_path: str) -> None: ...
    @staticmethod
    def to_biotite(protein: Protein) -> AtomArray: ...

# protein_frames.py - Frame calculations
class ProteinFrameBuilder:
    @staticmethod
    def to_bb_rigid(protein: Protein) -> Rigid: ...
    @staticmethod
    def to_psi_sin_cos(protein: Protein) -> np.ndarray: ...

# protein_transform.py - Coordinate transforms
class ProteinTransform:
    @staticmethod
    def center(protein: Protein) -> Protein: ...
```

---

### 2. Extract Orphaned Functions into Classes

**Problem:** Transformation functions exist outside classes in `rigid.py` and `utils.py`.

**Solution:** Create dedicated builder classes.

#### Create `RigidBuilder` class (from `rigid.py`):

```python
# rigid_builder.py
class RigidBuilder:
    @staticmethod
    def from_matrix_global_to_local(matrix: np.ndarray) -> Rigid:
        """Converts 3D position matrix to Rigid transformation."""
        ...

    @staticmethod
    def from_matrix_local_to_global(matrix: np.ndarray) -> Rigid:
        """Wrapper that inverts global-to-local transformation."""
        ...
```

#### Create `BackboneBuilder` class (from `utils.py`):

```python
# backbone_builder.py
class BackboneBuilder:
    @staticmethod
    def calculate_backbone(rigid: Rigid, psi_angles: np.ndarray) -> np.ndarray:
        """Reconstruct backbone from rigid frames and dihedral angles."""
        ...

    @staticmethod
    def psi_angles_to_rotation(psi_sin_cos: np.ndarray) -> Rotation:
        """Convert dihedral angles to rotation matrices."""
        ...
```

---

### 3. Remove Duplicate Code

**File:** `rotation_testing.py` (106 lines)

**Problem:** Contains `rotate_vectors()` and `rotate_vectors_matmul()` which duplicate functionality already in `Rotation.apply()`.

**Solution:**
- Delete `rotation_testing.py` entirely, OR
- Consolidate into the `Rotation` class as alternative implementations if needed for benchmarking

---

### 4. Consolidate Magic Constants

**Problem:** Default frame transformations are hardcoded in multiple places:
- `protein.py` lines 169-179
- `utils.py` lines 38-43

**Solution:** Create a single source of truth in `protein_constants.py`:

```python
# protein_constants.py
class DefaultFrames:
    """Default rigid frame transformations for backbone construction."""

    ROT = np.array([
        [-0.5272, 0.0000, 0.8497],
        [0.0000, 1.0000, 0.0000],
        [-0.8497, 0.0000, -0.5272]
    ])

    TRANS = np.array([1.4613, 0.0, 0.0])
```

---

## Medium Priority Refactoring

### 5. Implement `RotFlow` Class

**File:** `flow_testing/flow/rot_flow.py` (0 lines - empty stub)

**Action:** Implement rotation space flow matching following the pattern in `r3_flow.py`.

---

### 6. Reorganize `utils.py`

**Problem:** Mixed concerns - backbone reconstruction and frame utilities combined.

**Solution:** Split into focused modules:

```
utils.py (60 lines) →
├── backbone_builder.py  (backbone reconstruction logic)
└── frame_builder.py     (frame-related utilities)
```

---

## Proposed New Directory Structure

```
flow_testing/data/
├── __init__.py
├── protein.py              (slimmed down - data only)
├── protein_io.py           (NEW - I/O operations)
├── protein_frames.py       (NEW - frame calculations)
├── protein_transform.py    (NEW - coordinate transforms)
├── rigid.py                (simplified - class only)
├── rigid_builder.py        (NEW - transformation builders)
├── rot.py                  (unchanged)
├── backbone_builder.py     (NEW - from utils.py)
├── frames.py               (NEW - default frame constants)
└── protein_constants.py    (expanded with DefaultFrames)

flow_testing/flow/
├── __init__.py
├── base.py                 (unchanged)
├── r3_flow.py              (unchanged)
└── rot_flow.py             (implement!)
```

---

## Quick Wins

These changes provide immediate benefit with minimal effort:

| Action | Files Affected | Benefit |
|--------|---------------|---------|
| Delete `rotation_testing.py` | 1 file | Remove 106 lines of duplicate code |
| Extract `DefaultFrames` constant class | 2 files | Single source of truth |
| Move I/O methods to `ProteinIO` | 1 file | Clearer responsibilities |

---

## Current Anti-Patterns Identified

1. **God Class** - `Protein` has too many responsibilities
2. **Duplicate Code** - `rotation_testing.py` duplicates `Rotation.apply()`
3. **Orphaned Functions** - Transformation functions exist outside classes
4. **Magic Constants** - Hardcoded arrays duplicated across files
5. **Empty Stub** - `rot_flow.py` is unimplemented

---

## Current Good Patterns (Keep These)

- Class-based design for core data structures (`Protein`, `Rigid`, `Rotation`)
- Composition over inheritance in `Rigid` (contains `Rotation` object)
- Abstract base classes for extensibility (`Sampleable`, `Alpha`, `Beta`, `Flow`)
- Method chaining patterns (e.g., `protein.center().to_bb_rigid()`)
- Factory methods for construction (`from_pdb()`, `from_biotite()`)
- Type hints throughout the codebase
- Decoupled data and flow modules
