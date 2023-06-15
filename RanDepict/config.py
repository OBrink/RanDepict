from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
from typing import List, Optional

@dataclass
class RandomDepictorConfig:
    """
    Examples
    --------
    >>> c1 = RandomDepictorConfig(seed=24, styles=["cdk", "indigo"])
    >>> c1
    RandomDepictorConfig(seed=24, hand_drawn=False, augment=True, styles=['cdk', 'indigo'])
    >>> c2 = RandomDepictorConfig(styles=["cdk", "indigo", "pikachu", "rdkit"])
    >>> c2
    RandomDepictorConfig(seed=42, hand_drawn=False, augment=True, styles=['cdk', 'indigo', 'pikachu', 'rdkit'])
    """
    seed: int = 42
    hand_drawn: bool = False
    augment: bool = True
    # unions of containers are not supported yet
    # https://github.com/omry/omegaconf/issues/144
    # styles: Union[str, List[str]] = field(default_factory=lambda: ["cdk", "indigo", "pikachu", "rdkit"])
    styles: List[str] = field(default_factory=lambda: ["cdk", "indigo", "pikachu", "rdkit"])

    @classmethod
    def from_config(cls, dict_config: Optional[DictConfig] = None) -> 'RandomDepictorConfig':
        return OmegaConf.structured(cls(**dict_config))

    def __post_init__(self):
        # Ensure styles are always List[str] when "cdk, indigo" is passed
        if isinstance(self.styles, str):
            self.styles = [v.strip() for v in self.styles.split(",")]
        if len(self.styles) == 0:
            raise ValueError("Empty list of styles was supplied.")
        # Not sure if this is the best way in order to not repeat the list of styles
        ss = set(self.__dataclass_fields__['styles'].default_factory())
        if any([s not in ss for s in self.styles]):
            raise ValueError(f"Use only {', '.join(ss)}")
