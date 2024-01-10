import pathlib
from copy import deepcopy
from typing import List, Optional, Type, TypeVar
from dataclasses import dataclass

PROMPTS_ROOT = (pathlib.Path(__file__).parent).resolve()

T = TypeVar("T")

@dataclass(frozen=True)
class Item:
    node_description: str
    neighbor_text: List[str]
    id: Optional[str] = None
    content: Optional[str] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Item from dict.")
        id = data.pop("id", None)
        return cls(**dict(data, id=id))
    
def get_prompt(
    task_name: str,
    task_item: Item,
):
    if task_name == "final_process":
        prompt_filename = "final_node.prompt"
    elif task_name == "mid_process":
        prompt_filename = "mid_node.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    
    node_description = task_item.node_description
    neighbor_text = task_item.neighbor_text

    # Format the potential categories into strings
    formatted_neighbor_texts = []
    for neighbor_index, neighbor in enumerate(neighbor_text ):
        formatted_neighbors = f"Neighbor [{neighbor_index+1}]({neighbor}) "
        formatted_neighbor_texts.append(
            formatted_neighbors
        )

    if task_name == "final_process":
        return_node_text = prompt_template.format(
                node_description=node_description,
                neighbor_text="\n".join(formatted_neighbor_texts),
                )
    elif task_name == "mid_process":
         return_node_text = prompt_template.format(
            node_description=node_description,
            neighbor_text=" ".join(formatted_neighbor_texts),
            )
    return return_node_text