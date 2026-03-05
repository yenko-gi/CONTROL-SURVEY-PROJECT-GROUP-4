import math
import numpy as np
import matplotlib.pyplot as plt

def dms_to_decimal(d, m, s):
    return d + m/60.0 + s/3600.0

def get_float(p):
    return float(input(p))

def get_dms(p):
    d, m, s = map(float, input(f"{p} (D M S): ").split())
    return dms_to_decimal(d, m, s)

def compute_provisional():
    print("\n=== PROVISIONAL COORDINATES ===")
    name = input("Station name: ").strip() or "P"
    
    print("\nStation A:"); N_A = get_float("  N: "); E_A = get_float("  E: ")
    print("\nStation B:"); N_B = get_float("  N: "); E_B = get_float("  E: ")
    
    if input("Have bearings? (y/n): ").lower() == 'y':
        brg_A = get_dms("Bearing A→P")
        brg_B = get_dms("Bearing B→P")
    else:
        dN, dE = N_B - N_A, E_B - E_A
        brg_AB = math.degrees(math.atan2(dE, dN)) % 360
        brg_BA = (brg_AB + 180) % 360
        brg_A = (brg_AB - get_dms("Angle at A")) % 360
        brg_B = (brg_BA + get_dms("Angle at B")) % 360
    
    dN, dE = N_B - N_A, E_B - E_A
    sin_diff = math.sin(math.radians(brg_B - brg_A))
    if abs(sin_diff) < 1e-12:
        print("Error: Parallel"); return None, None, None
    
    dAP = (dN * math.sin(math.radians(brg_B)) - dE * math.cos(math.radians(brg_B))) / sin_diff
    brg_A_rad = math.radians(brg_A)
    N_P = N_A + dAP * math.cos(brg_A_rad)
    E_P = E_A + dAP * math.sin(brg_A_rad)
    
    print(f"\n✅ Provisional {name}: N={N_P:.3f}, E={E_P:.3f}")
    return N_P, E_P, name

def compute_cut(N_C, E_C, N_P, E_P, brg_deg):
    """
    ORIGINAL coordinate-cut formulas (correct for this method).
    cut_N = cot(bearing) * ΔE - ΔN
    cut_E = tan(bearing) * ΔN - ΔE
    """
    brg_rad = math.radians(brg_deg)
    delta_N = N_P - N_C
    delta_E = E_P - E_C
    sin_b = math.sin(brg_rad)
    cos_b = math.cos(brg_rad)
    
    # Handle cardinal directions
    eps = 1e-12
    
    if abs(sin_b) < eps:  # 0° or 180°
        cut_N = float('inf')
        cut_E = float('inf')
    elif abs(cos_b) < eps:  # 90° or 270°
        cut_N = float('inf')
        cut_E = float('inf')
    else:
        cot_b = cos_b / sin_b
        tan_b = sin_b / cos_b
        cut_N = cot_b * delta_E - delta_N
        cut_E = tan_b * delta_N - delta_E
    
    s1 = delta_E / sin_b if abs(sin_b) > eps else float('inf')
    
    return cut_N, cut_E, s1

def least_squares_adjust(stations, N_P, E_P):
    """
    CORRECTED: Use least squares to find best intersection.
    Each station defines a line: cE - tan(bearing)*cN = cut_E
    """
    n = len(stations)
    A = []
    b = []
    
    for N_C, E_C, brg in stations:
        brg_rad = math.radians(brg)
        sin_b = math.sin(brg_rad)
        cos_b = math.cos(brg_rad)
        
        # Skip if bearing is exactly cardinal (degenerate case)
        if abs(cos_b) < 1e-12:
            # For 90°/270°, use constraint: cN = N_C - N_P
            A.append([1, 0])
            b.append(N_C - N_P)
        elif abs(sin_b) < 1e-12:
            # For 0°/180°, use constraint: cE = E_C - E_P
            A.append([0, 1])
            b.append(E_C - E_P)
        else:
            tan_b = sin_b / cos_b
            delta_N = N_P - N_C
            delta_E = E_P - E_C
            cut_E = tan_b * delta_N - delta_E
            
            # Line: -tan(bearing)*cN + 1*cE = cut_E
            A.append([-tan_b, 1])
            b.append(cut_E)
    
    A = np.array(A)
    b = np.array(b)
    
    solution = np.linalg.lstsq(A, b, rcond=None)[0]
    return solution[0], solution[1]

def plot_results(stations, cuts, N_P, E_P, cN, cE, name):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(stations)))
    
    for i, ((N_C, E_C, brg), (cut_N, cut_E, s1), color) in enumerate(zip(stations, cuts, colors)):
        if abs(cut_N) < 1e6 and abs(cut_E) < 1e6:
            ax.plot(cut_E, cut_N, 'o', color=color, markersize=10, label=f"C{i+1}")
            ax.plot([0, cut_E], [0, cut_N], '--', color=color, alpha=0.5)
            ax.annotate(f"C{i+1}\nΔN={cut_N:.3f}\nΔE={cut_E:.3f}", 
                       (cut_E, cut_N), xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.plot(cE, cN, 'D', color='purple', markersize=12, label='Adjusted')
    ax.annotate(f"ADJUSTED\nΔN={cN:.3f}\nΔE={cE:.3f}", 
               (cE, cN), xytext=(5, 5), textcoords='offset points',
               fontweight='bold', color='purple',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    for cut_N, cut_E, s1 in cuts:
        if abs(cut_N) < 1e6 and abs(cut_E) < 1e6:
            ax.plot([cut_E, cE], [cut_N, cN], ':', color='gray', alpha=0.3)
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('ΔEasting (m)')
    ax.set_ylabel('ΔNorthing (m)')
    ax.set_title(f'Cut Analysis - Station {name}')
    ax.legend(loc='best')
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

def main():
    print("="*60 + "\nSURVEYING CUT COMPUTATION (CORRECTED)\n" + "="*60)
    
    if input("Have provisional coordinates? (y/n): ").lower() == 'y':
        name = input("Station name: ").strip() or "P"
        N_P = get_float("Northing: ")
        E_P = get_float("Easting: ")
    else:
        N_P, E_P, name = compute_provisional()
        if N_P is None: return
    
    n = int(input("\nNumber of known stations: "))
    stations = []
    
    print("\n--- Enter station data ---")
    for i in range(n):
        print(f"\nStation {i+1}:")
        stations.append((get_float("  Northing: "), get_float("  Easting: "), get_dms("  Bearing")))
    
    print("\n" + "="*60 + "\nCUTS\n" + "="*60)
    cuts = []
    for i, (N_C, E_C, brg) in enumerate(stations):
        cut_N, cut_E, s1 = compute_cut(N_C, E_C, N_P, E_P, brg)
        cuts.append((cut_N, cut_E, s1))
        if abs(cut_N) < 1e6 and abs(cut_E) < 1e6:
            print(f"C{i+1}: cut_N = {cut_N:+.4f}, cut_E = {cut_E:+.4f}, s1 = {s1:.4f}")
        else:
            print(f"C{i+1}: cut undefined (cardinal bearing)")
    
    cN, cE = least_squares_adjust(stations, N_P, E_P)
    
    print("\n" + "="*60 + "\nADJUSTMENT\n" + "="*60)
    print(f"Provisional: N = {N_P:.4f}, E = {E_P:.4f}")
    print(f"Adjusted:    N = {N_P + cN:.4f}, E = {E_P + cE:.4f}")
    print(f"Corrections: ΔN = {cN:+.4f}, ΔE = {cE:+.4f}")
    
    plot_results(stations, cuts, N_P, E_P, cN, cE, name)
    print(f"\n✅ Complete: {name} = ({N_P + cN:.4f}, {E_P + cE:.4f})")

if __name__ == "__main__":
    main()