import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
from trace_gen import default_usable_states, simulate_trace, select_states, pause_state, power_law_pause_state, activity_state, jump_state, high_noise_pause_state, intro_lost_bead, intro_missing_regions, intro_high_noise, intro_inconsistent_noise, intro_unobservable_activity, intro_compaction_noise, generate_training_dataset, save_dataset

def plot_trace(trace, title='trace', show_clean=True, show_noisy=True, vis_state=None, line_registry=None, fig=None, ax=None):
    created_fig = False
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.subplots_adjust(right=0.82)
        created_fig = True
    lines = []
    labels = []
    registry = line_registry if line_registry is not None else {}
    if show_noisy and "with_noise" in trace:
        ln_noisy, = ax.plot(trace["time"], trace["with_noise"], label="with_noise", color="cornflowerblue", linewidth=0.8, alpha=0.8)
        lines.append(ln_noisy)
        labels.append("with_noise")
        registry.setdefault("with_noise", []).append(ln_noisy)
    if show_clean and "clean" in trace:
        ln_clean, = ax.plot(trace["time"], trace["clean"], label="clean", color="black", linewidth=1.2, alpha=1)
        lines.append(ln_clean)
        labels.append("clean")
        registry.setdefault("clean", []).append(ln_clean)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("extension / nt")
    ax.set_title(title)
    legend = ax.legend()

    if lines:
        for legline, origline in zip(legend.get_lines(), lines):
            legline.set_picker(True)
            legline.set_pickradius(5)
            if vis_state is not None:
                target_vis = vis_state.get(origline.get_label(), origline.get_visible())
                origline.set_visible(target_vis)
                legline.set_alpha(1.0 if target_vis else 0.2)
            elif origline.get_label() == "clean":
                origline.set_visible(False)
                legline.set_alpha(0.2)

        def on_pick(event):
            legline = event.artist
            try:
                idx = legend.get_lines().index(legline)
            except ValueError:
                return
            label = lines[idx].get_label()
            current_vis = registry.get(label, [lines[idx]])[0].get_visible()
            new_vis = not current_vis
            for ln in registry.get(label, []):
                ln.set_visible(new_vis)
            if vis_state is not None:
                vis_state[label] = new_vis
            for ax_other in fig.axes:
                leg_other = ax_other.get_legend()
                if leg_other:
                    for legline_other in leg_other.get_lines():
                        if legline_other.get_label() == label:
                            legline_other.set_alpha(1.0 if new_vis else 0.2)
            legend.set_alpha(1.0)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", on_pick)

def demo_bad_features():
    state_names = ["pause_fast", "pause_slow", "pause_power", "activity_up", "activity_down"]
    states = select_states(default_usable_states(), include=state_names)
    
    good_trace = simulate_trace(total_time=200.0, dt=0.02, states=states, background_noise_std=10.0, seed=42)
    
    trace_lost_bead = intro_lost_bead(good_trace.copy(), loss_time=120.0, seed=42)
    trace_missing = intro_missing_regions(good_trace.copy(), n_regions=2, seed=42)
    trace_high_noise = intro_high_noise(good_trace.copy(), noise_multiplier=3.0, seed=42)
    trace_inconsistent = intro_inconsistent_noise(good_trace.copy(), n_transitions=3, seed=42)
    trace_unobservable = intro_unobservable_activity(good_trace.copy(), noise_multiplier=5.0, seed=42)
    trace_compaction = intro_compaction_noise(good_trace.copy(), noise_std_range=(30.0, 5.0), seed=42)
    
    fig, axs = plt.subplots(7, 1, figsize=(12, 16), sharex=True)
    fig.subplots_adjust(right=0.82, hspace=0.4)
    
    plot_trace(good_trace, title="Good trace (baseline)", fig=fig, ax=axs[0])
    plot_trace(trace_lost_bead, title="Lost bead (high noise after loss)", fig=fig, ax=axs[1])
    plot_trace(trace_missing, title="Missing regions (NaN gaps)", fig=fig, ax=axs[2])
    plot_trace(trace_high_noise, title="High noise (3x multiplier)", fig=fig, ax=axs[3])
    plot_trace(trace_inconsistent, title="Inconsistent noise levels (unwanted for polymerase)", fig=fig, ax=axs[4])
    plot_trace(trace_compaction, title="Compaction noise (decreasing, desired for compaction traces)", fig=fig, ax=axs[5])
    plot_trace(trace_unobservable, title="Unobservable activity (excessive noise + drift)", fig=fig, ax=axs[6])
    
    plt.show()

def main():
    state_names = ["pause_fast", "pause_slow", "pause_power", "activity_down"]
    states = select_states(default_usable_states(), include=state_names)

    states_alt = [
        pause_state("pause_fast", rate_range=(1.0, 3.0), weight=0.4),
        pause_state("pause_slow", rate_range=(0.2, 0.7), weight=0.3),
        power_law_pause_state("pause_power", alpha=2.0, min_pause=0.5, max_pause=8.0, weight=0.1),
        activity_state("activity_up", slope_range=(30.0, 60.0), duration_range=(0.5, 3.0), weight=0.15, direction=1),
        jump_state("jump_back", jump_range=(50.0, 600.0), weight=0.001, direction=-1),
        high_noise_pause_state("high_noise", rate_range=(0.5, 1.0), noise_std_range=(10.0, 40.0), weight=0.002),
    ]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(right=0.82)
    vis_state = {"with_noise": True, "clean": False}
    line_registry = {'with_noise': [], 'clean': []}

    def draw_traces(event=None):
        trace = simulate_trace(total_time=200.0, dt=0.02, states=states, background_noise_std=10.0, seed=random.randint(1, 1000000))
        trace_alt = simulate_trace(total_time=200.0, dt=0.02, states=states_alt, background_noise_std=8.0, seed=random.randint(1, 1000000))
        for ax in axs:
            ax.cla()
        line_registry["with_noise"].clear()
        line_registry["clean"].clear()
        plot_trace(trace, title="selected states trace", fig=fig, ax=axs[0], vis_state=vis_state, line_registry=line_registry)
        plot_trace(trace_alt, title="states_alt trace", fig=fig, ax=axs[1], vis_state=vis_state, line_registry=line_registry)
        fig.canvas.draw_idle()

    def draw_traces_initial(event=None):
        trace = simulate_trace(total_time=165.)

    draw_traces()

    button_ax = fig.add_axes([0.88, 0.02, 0.1, 0.05])
    button = Button(button_ax, "regen")
    button.on_clicked(draw_traces)

    plt.show()

def demo_dataset_generation():
    print("generating test dataset (10,000 traces)...")
    dataset = generate_training_dataset(
        n_traces=10_000,
        usable_ratio=0.5,
        total_time_range=(100.0, 300.0),
        dt_range=(0.01, 0.05),
        noise_std_range=(5.0, 20.0),
        n_workers=None,
        base_seed=42,
        chunk_size=1000
    )
    
    print(f"done: {dataset['n_traces']} traces ({dataset['n_usable']} usable, {dataset['n_unusable']} unusable)")
    print()
    for i in range(min(5, len(dataset['traces']))):
        label = "usable" if dataset['labels'][i] == 1 else "unusable"
        bad_feat = dataset['metadata'][i].get('bad_feature_type', 'N/A')
        print(f"  trace {i}: {label}, bad_feature={bad_feat}, len={len(dataset['traces'][i])}")
    
    return dataset

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo_bad":
        demo_bad_features()
    elif len(sys.argv) > 1 and sys.argv[1] == "demo_dataset":
        demo_dataset_generation()
    else:
        main()
