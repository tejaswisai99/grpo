import webshop_interaction as ws

def run_many(goal_start:int, goal_end:int) -> None:
    for goal_num in range(goal_start, goal_end):
        goal = ws.initialize_goal(goal_num, "text")
        print(goal)


if __name__ == "__main__":
    run_many(goal_start=0, goal_end=1)
