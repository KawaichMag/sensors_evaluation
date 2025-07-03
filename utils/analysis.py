import os

from objects.Types import Fitness

def get_solutions(file_path: str) -> list[Fitness]:
    with open(file_path, "r") as f:
        solutions = []
        for line in f:
            solution = line.split(": ")[1].split(" ")
            solutions.append((float(solution[0]), float(solution[1])))
        return solutions

def save_interest_solutions(file_path: str) -> None:
    solutions = get_solutions(file_path)    
    interest = (solutions[0], solutions[len(solutions) // 2], solutions[-1])

    if not os.path.exists("analysis"):
        os.makedirs("analysis")

    with open("analysis/solutions.txt", "a") as f:
        for solution in interest:
            f.write(f"{solution[0]} {solution[1]} ")
