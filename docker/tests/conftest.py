"""docker/tests 공용 설정: docker/ 폴더를 sys.path 맨 앞에 넣는다(모든 테스트가 `import nmfc_*`에 의존한다)."""
import sys
from pathlib import Path

_DOCKER_DIR = str(Path(__file__).resolve().parent.parent)
if _DOCKER_DIR in sys.path:
    sys.path.remove(_DOCKER_DIR)
sys.path.insert(0, _DOCKER_DIR)
