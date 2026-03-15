"""Launch the live webcam exercise tracker."""

from __future__ import annotations

from repcounter.live import LiveConfig, LiveExerciseTracker, build_arg_parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    tracker = LiveExerciseTracker(
        LiveConfig(
            source=args.source,
            exercise_model_path=args.exercise_model,
            stage_model_path=args.clf or args.stage_model,
            reps_per_set=args.reps_per_set,
            target_sets=args.target_sets,
            min_exercise_confidence=args.min_exercise_confidence,
            min_stage_confidence=args.min_stage_confidence,
            exercise_window=args.exercise_window,
            exercise_min_votes=args.exercise_min_votes,
            stage_window=args.stage_window,
            stage_min_votes=args.stage_min_votes,
            rep_cooldown_frames=args.rep_cooldown_frames,
            rep_feedback_seconds=args.rep_feedback_seconds,
            lock_exercise=args.lock_exercise,
            show_debug=args.show_debug,
        )
    )
    tracker.run()


if __name__ == "__main__":
    main()
