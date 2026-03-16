from pydantic import BaseModel


class PromptStrings(BaseModel):
    role_prompt: str
    pre_solution_prompt_template: str
    code_structure_intro: str
    crossover_operator: str
    selected_solutions_intro: str
    selected_solution_intro: str
    mutation_operator: str
    population_summary_intro: str
    candidate_code_wrapper: str
    no_code_exception_msg: str
    general_error_msg: str
    candidate_prompt_template: str
    bo_lib_prompt_cpu: str
    bo_lib_prompt_gpu: str
    bo_task_prompt_template: str
    general_task_prompt: str
    output_format_prompt: str
    code_structure_template: str
    mini_bo_code_structure_template: str
    bo_code_structure_template: str
    time_prompt_template: str
    main_aoc_prompt_template: str
