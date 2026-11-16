# Do this import to ensure that the Gym environments get registered properly
import deep_sprl.environments  # noqa: F401
from .abstract_experiment import AbstractExperiment, CurriculumType
from .colored_sokoban_experiment import ColoredSokobanExperiment
from .button_maze_experiment import ButtonMazeExperiment

__all__ = ['AbstractExperiment', 'CurriculumType', 'ColoredSokobanExperiment', 'ButtonMazeExperiment']
