import optuna
import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter
import tiktoken

from .bootstrap import BootstrapFewShot

class OptunaJointCostOptimizer(Teleprompter):
    def __init__(self, bootstrap_metric, optim_metric=None, teacher_settings={}, min_demos=1, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, num_candidate_programs=16, num_threads=6):
        self.bootstrap_metric = bootstrap_metric
        if optim_metric is None:
            self.optim_metric = bootstrap_metric
        else:
            self.optim_metric = optim_metric
        self.teacher_settings = teacher_settings
        self.max_rounds = max_rounds

        self.num_threads = num_threads

        # Load the tiktoken encoding for gpt-4
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        assert self.encoding.decode(self.encoding.encode("hello world")) == "hello world"

        self.min_num_samples = min_demos
        self.max_num_samples = max_bootstrapped_demos
        self.num_candidate_sets = num_candidate_programs
        # self.max_num_traces = 1 + int(max_bootstrapped_demos / 2.0 * self.num_candidate_sets)

        # Semi-hacky way to get the parent class's _bootstrap function to stop early.
        # self.max_bootstrapped_demos = self.max_num_traces
        self.max_labeled_demos = max_labeled_demos

        print("Going to find cost-efficient traces per predictor. Going to select between", self.min_num_samples, "and", self.max_num_samples, "bootstrap demos per predictor.")
        # print("Going to sample", self.max_num_traces, "traces in total.")
        print("Will attempt to train", self.num_candidate_sets, "candidate sets during joint cost optimization.")

    def objective(self, trial):
        program2 = self.student.reset_copy()

        is_format_instructions = trial.suggest_int("formatting_instructions_added", 0, 1)
        if is_format_instructions == 1:
            dspy.settings.show_guidelines = True
            print(dspy.settings.show_guidelines)
        else:
            dspy.settings.show_guidelines = False
            print(dspy.settings.show_guidelines)

        index = 0
        for (name, compiled_predictor), (_, program2_predictor) in zip(self.compiled_teleprompter.named_predictors(), program2.named_predictors()):
            # find optimal temperature for each predictor
            program2_predictor.config["temperature"] = trial.suggest_float(f"temperature_for_predictor_{name}", 0.0, 0.6, step=0.2)

            # find optimal set of demos for each predictor
            if index < len(self.compiled_teleprompter.named_predictors())-1:
                all_demos = [demo for demo in compiled_predictor.demos if 'augmented' in demo]
            else:
                all_demos = compiled_predictor.demos
            demo_indices = []
            num_demos = trial.suggest_int(f"num_demos_for_predictor_{name}", self.min_num_samples, min(self.max_num_samples, len(all_demos)))
            for idx in range(num_demos):
                if idx == num_demos-1:
                    import itertools
                    demo_combinations = [list(x) for x in list(itertools.combinations(list(range(len(all_demos))), num_demos))]
                    next_demo_indices = trial.suggest_categorical(f"selected_{idx+1}_demo_indices_for_predictor_{name}", demo_combinations)
                    demo_indices += next_demo_indices
            
            program2_predictor.demos = [all_demos[i] for i in demo_indices]
            index += 1

        evaluate = Evaluate(devset=self.valset, metric=self.optim_metric, num_threads=self.num_threads,
                            display_table=False, display_progress=True)
        score, _ = evaluate(program2, return_all_scores=True)

        # get the total number of tokens of the demos included in the program
        def get_total_demo_tokens(compiled_program):
            total_prompt_tokens = 0
            for _, predictor in compiled_program.named_predictors():
                input_fields = predictor.signature.input_fields
                demos = predictor.demos
                for demo in demos:
                    for key, value in demo.items():
                        if key in input_fields:
                            # print("key: ", key, "value: ", value)
                            total_prompt_tokens += len(self.encoding.encode(str(value)))
            return total_prompt_tokens

        trial.set_user_attr("program", program2)
        return score, get_total_demo_tokens(program2)


    def compile(self, student, *, teacher=None, trainset, valset=None):
        self.trainset = trainset
        self.valset = valset or trainset
        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher is not None else student.reset_copy()
        teleprompter_optimize = BootstrapFewShot(metric=self.bootstrap_metric, 
                                                 max_bootstrapped_demos=len(trainset),
                                                max_labeled_demos=0,
                                                teacher_settings=self.teacher_settings,
                                                max_rounds=self.max_rounds)
        print("Start finding bootstrapped demos for each predictor...")
        self.compiled_teleprompter = teleprompter_optimize.compile(self.student, teacher=self.teacher, trainset=self.trainset)
        
        # specify optuna study with multi-obj and median pruner
        median_pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(directions=["maximize", "minimize"], pruner=median_pruner)
        study.optimize(self.objective, n_trials=self.num_candidate_sets)

        # get all programs and pareto_efficient programs
        programs = study.trials
        pareto_efficient_programs = study.best_trials
        print('Done!')
        print(f"Nr of pareto-efficient programs: {len(study.best_trials)}/{len(study.trials)}")
        return programs, pareto_efficient_programs