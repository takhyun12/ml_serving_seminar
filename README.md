# 고성능 ML 서빙 최적화

현재 라스코에서 운영중인 AI 백엔드 서버들은 Python을 기반으로 구축되어 있습니다.

이러한 Python 기반의 시스템은 통상적으로 다른 언어에 비해 실행 속도가 느리다고 알려져 있는데요, 
본 세미나에서는 이러한 Python 기반의 시스템에 대해 자세히 알아보고 그 과정에서 일부 잘못된 오해와 편견 그리고 성능을 개선하기 위한 다양한 방법을 고찰하고자 합니다.

세미나를 통해 새로이 알게되는 내용이 실제 서비스에 practical 하게 접목 될 수 있도록 하기 위해 최대한 예재를 기반으로 준비하였습니다.

## Table of Contents
- **백엔드 동작구조에 대한 이해** - `node.js`와 `fastapi`의 동작방식에 관한 고찰
- **ML 서빙 최적화** - Infra-level에서의 최적화 기법 고찰
- **ML 서빙 최적화** - Code-level에서의 최적화 기법 고찰

## 백엔드 동작구조에 대한 이해

### Node.JS (Javascript)
Python 기반 백엔드 시스템의 동작구조를 알아보기에 앞서, 배경지식을 쌓는 차원에서 Node.js에 대해서 먼저 알아보도록 하겠습니다.
Node.js는 Javascript 기반의 런타임 프레임워크이며, 현재 라스코에서도 일부 사용 중이므로 알아두면 좋을 것 같아 준비하였습니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/b3a2d7df-54f0-4404-945a-1541b3fe84a3" width="800" alt=""/>

**[그림 1]** Node.js 런타임 환경의 동작 구조

###

우리가 Node.js 런타임 환경에 코드를 작성하게 되면, [그림 1]과 같은 형태의 application level에서 동작하게 됩니다. 
이를 자세히 살펴보면, 동적으로 생성한 데이터들을 보관하는 heap과 함수의 순서를 저장하는 call stack이 있는 것을 볼 수 있습니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/57f3fff4-6b24-42c8-8f3b-41238cc57148" width="800" alt=""/>
<img src="https://media.oss.navercorp.com/user/37614/files/9b7ca60f-a560-4f7e-bdf2-7b196fd163c0" width="800" alt=""/>

**[그림 2]** Node.js 런타임 환경에서의 call stack의 동작 구조

###

먼저 call stack에 대해서 살펴보도록 하겠습니다.

만약 코드에서 main() 이라는 함수가 실행되었고, main 함수 내에 first() 라는 함수와 second() 라는 함수가 있다면, 
[그림 2]와 같은 FILO(First In Last Out)의 형태로 동작하게 됩니다.

즉, 함수의 실행 순서대로 call stack에 쌓이게 되며, 각 함수가 종료된 시점(return)에서 하나씩 call stack에서 제거되는 단순한 구조를 갖습니다.
이러한 방식을 통해 함수가 호출된 순차 흐름을 기억하게 되고, return 되어야 하는 위치를 알 수 있게 됩니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/3e252ea9-6357-44d0-ba48-4d1b32da9209" width="800" alt=""/>

**[그림 3]** Node APIs를 사용하는 경우 Non-blocking I/O를 설명하기 위한 그림

###

그렇다면, 조금 더 깊게 들어가서 node.js에서 기본적으로 제공하는 함수를 사용하게 되면 어떠한 구조로 동작하는지 알아보도록 하겠습니다.
node.js에서 기본적으로 제공하는 함수는 node.js 커뮤니티에서는 `node.js API` 라고 부르고 있는 점을 참고 바랍니다.

예를 들어서, [그림 3]와 같이 `node.js API` 중 하나인 `setTimeout` 함수를 사용하였다고 가정해보도록 하겠습니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/60fe8d88-8c05-4f62-8440-aeed31ea093f" width="800" alt=""/>

**[그림 4]** `node.js API`가 실행되는 구조

###

이러한 `node.js API`가 호출되면, node.js에서는 non-blocking event driven 방식이기 때문에 main thread에서 이러한 처리를 직접 수행하지 않고,
별도의 thread를 할당하여 작업을 수행하게 됩니다.

이때, node.js app은 별도의 thread 할당이 완료된 시점에서 call stack 해당 작업을 stack에서 제외하게 되는데, 이러한 처리 방식을 non-blocking I/O라고 합니다.
쉽게 말해, call stack을 점유하고 있지 않고 바로 다음 작업을 실행할 수 있는 상태가 되는 것입니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/6de5462d-547c-4e6c-80d5-139955ed2834" width="800" alt=""/>
<img src="https://media.oss.navercorp.com/user/37614/files/146611a8-53fa-4b04-a3ef-846f4a066774" width="800" alt=""/>

**[그림 5]** `node.js API`와 `event loop`의 동작구조

###

할당된 작업을 마친 thread는 호출해야 하는 callback을 task queue라는 일종의 대기줄에 할당하게 되며,
call stack이 비어 있는 시점에서 node.js의 event loop에 의해서 call stack으로 반환되게 됩니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/874c45bc-4716-4ddd-8703-8b18c71c35e7" width="800" alt=""/>

**[그림 6]** node.js 런타임 프레임워크에 대한 요약

###

이처럼 node.js는 main application이 single thread로 동작하며, node.js api를 통해 multi thead로 작업을 할당해서 처리하게 됩니다. 
즉, single thread는 작고 가벼운 일만을 수행하기 때문에 I/O의 측면에서는 매우 효율적이며 이벤트 단위로 처리하기 때문에 event-driven 방식이라고도 합니다.

node.js 개발자들은 이러한 특징에 따라 single thread로 동작하는 event loop에 heavy calculation을 할당하지 않으며, 
image resizing, video encoding 등의 작업은 worker thread로 할당하는 방식으로 시스템을 최적화하고 있습니다.

###

### FastAPI (Python)
이번에는 Python 커뮤니티에서 대표적으로 사용되고 있는 FastAPI 프레임워크의 동작구조에 대해서 살펴보도록 하겠습니다. 

###

<img src="https://media.oss.navercorp.com/user/37614/files/60a1ee12-eb0e-4c27-ad54-1de17fe01227" width="800" alt=""/>

**[그림 7]** 다양한 백엔드 서버의 벤치마크 결과

###

FastAPI에 대해 자세히 살펴보기 전 다양한 백엔드 서버의 벤치마크를 비교한 그림을 살펴보도록 하겠습니다.
FastAPI의 성능의 측면에서 node.js보다 훨씬 앞서고, Golang과 거의 유사한 성능임을 볼 수 있습니다.

즉, Python 기반의 시스템이 무조건 느리다는 것은 잘못 알려진 사실이며, 성능 자체는 오히려 매우 좋은 편임을 볼 수 있습니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/a26246ba-361f-466b-be95-83872b0251c2" width="800" alt=""/>

**[그림 8]** FastAPI 프레임워크의 구조

###

그렇다면, 이처럼 FastAPI의 성능이 좋은 이유에 대해서 자세히 살펴보도록 하겠습니다. 
FastAPI의 구조는 [그림 8]과 같이 Starlette > Uvicorn > Uvloop로 이어지는 연쇄 구조로 구성되어 있습니다.

이러한 구조에서 `Uvloop`는 Node.js 비동기 I/O의 핵심인 `libuv`와 동일하며 이를 통해 Non-blocking I/O를 수행하게 됩니다.
또한, 이러한 구조가 Pure Python이 아닌 Cython으로 작성되어 빌드되었기 때문에 더욱 높은 성능을 보이게 됩니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/874c45bc-4716-4ddd-8703-8b18c71c35e7" width="914" alt=""/>
<img src="https://media.oss.navercorp.com/user/37614/files/a26246ba-361f-466b-be95-83872b0251c2" width="800" alt=""/>

**[그림 9]** node.js와 fastapi 구조의 비교 도표

###

[그림 9]와 같이 Node.js와 FastAPI를 비교하게 되면, Node.js의 Event loop와 동일한 역할을 수행하는 `uvloop`를 볼 수 있습니다.
이는, 기존 Node.js의 Event loop보다도 훨씬 성능이 좋은 것으로 알려져 있습니다.

또한, Cython으로 빌드된 `libuv`가 Node.js의 Non-blocking I/O의 역할을 수행하는 것을 볼 수 있는데,
실질적으로 두 프레임워크의 동작구조는 거의 같으며 단순히 FastAPI가 C언어를 기반으로 최적화가 조금 더 잘 되어 있는 정도의 차이입니다.

즉, FastAPI는 Python 코드를 기반으로 작성되지만, 사실상 C언어 기반의 시스템에 가깝다고도 볼 수 있습니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/7b5c74cd-26c3-4f4d-b05a-37928a5c9377" width="800" alt=""/>

**[그림 10]** FastAPI 런타임의 작업 할당 구조

###

다만, 이 부분에서 재미있는 특징으로 FastAPI는 node.js와 달리 멀티 쓰레딩이 아닌 멀티 프로세싱을 활용하고 있다는 점이 있습니다.

Python은 근본적인 문제인 GIL(Global Interpreter Lock)에 의해 multi-thread 에서의 성능이 크게 저하되기 때문에,
FastAPI에서는 [그림 10]과 같이 worker를 배정하는 과정에서 multi-process 방식을 사용하는 것을 볼 수 있습니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/a8f82cac-5f45-459f-ac65-48ef25ff27e3" width="800" alt=""/>

**[그림 11]** GIL(Global Interpreter Lock)

###

후술할 내용의 이해를 돕기위해 GIL(Global Interpreter Lock)에 대해서 조금 더 자세히 짚고 넘어가도록 하겠습니다.
GIL이란 python의 object의 동시 접근을 보호하기 위한 일종의 Mutex로서,
하나의 프로세스 내에서 한 시점에는 하나의 thread에 의해서만 접근할 수 있도록 하는 일종의 보호장치를 의미합니다.

즉, 하나의 값에 여러 쓰레드가 동시에 접근함으로써 값이 올바르지 않게 사용되는 문제를 방지하기 위한 장치입니다.

이러한 보호 장치가 필요한 이유를 이해하기 위해서는 Cpython이 동작하는 방식에 대해서 조금 더 자세히 알 필요가 있습니다.

먼저 Python에서 모든 것은 객체(Object)입니다. 
단순한 문자열 변수부터 리스트, 튜플, 딕셔너리와 같은 자료구조 심지어는 Class 까지 모두 Object 입니다.

이러한 Object는 `reference count`라는 값을 반드시 가지고 있는데, 이 값은 해당 객체를 가리키는 참조의 갯수를 의미합니다.

###

```python
import sys

a = [1, 2, 3, 4]

print(f'reference count : {sys.getrefcount(a)}')

b = a

print(f'reference count : {sys.getrefcount(a)}')

c = a

print(f'reference count : {sys.getrefcount(a)}')

c = None

print(f'reference count : {sys.getrefcount(a)}')

b = None

print(f'reference count : {sys.getrefcount(a)}')
```

```shell
reference count : 2
reference count : 3
reference count : 4
reference count : 3
reference count : 2
```

###

이러한 `reference count`는 `sys` 라이브러리의 `getrefcount` 함수를 사용하여 쉽게 확인할 수 있습니다. 

위 코드의 예재를 통해 살펴보면, 
`a = [1, 2, 3, 4]` 에서 `a`는 생성된 리스트에 대한 reference를 가지게 되어 count가 1이 되게 되고, 
`sys.getrefcount()`에 의해 호출되는 순간이 추가되어 참조 횟수가 2가 되는 것을 볼 수 있습니다.

그리고나서는 참조가 추가 될때마다 count가 증가하며, 변수의 참조가 끊어 지는 시점에서 count를 감소하게됩니다.
이러한 `reference count`는 0이 되는 순간 GC에서 해당 객체를 메모리에서 제거함으로써 완전히 소멸하게 됩니다.

이러한 특징에 따라 Python은 GIL이 도입되어 동시에 참조되는 것을 방지하게 됩니다.
동시에 이러한 `reference count`를 수정하는 것은 근본적으로 불가능하기 때문입니다.

따라서 Python 커뮤니티에서는 일반적으로 multi threading 환경에서의 성능이 보장되지 않기 때문에,
앞서 FastAPI의 예재와 같이 멀티 프로세싱의 개념을 적극 도입하여 사용하고 있습니다.

또한 이러한 GIL에 의한 문제를 개선하기 위해 수십 년간 다양한 시도를 통해 성능이 점차 개선되고 있으며,
최근에 들어서는 이전에 비해 비약적인 발전을 이루고 있습니다. 이 부분에 대해서는 추후 기회가 된다면 별도로 내용을 공유 드리도록 하겠습니다.

###

<img src="https://media.oss.navercorp.com/user/37614/files/330604f7-d106-443e-9b78-373fc4c62fc5" width="800" alt=""/>

**[그림 12]** ML 서빙 최적화 포인트

###

지금까지는 배경지식의 차원에서 FastAPI 구조에 대해서 가볍게 살펴보았는데요,
그렇다면 우리가 ML 서비스를 경량화하고 최적화해야 하기 위해 집중해야 부분은 어디일지 살펴보도록 하곘습니다.

[그림 12]는 Lasco.AI 처럼 ML 서비스 아키텍처에서 최적화 할 수 있는 포인트를 표현한 그림입니다. 

그림을 살펴보면, 우선적으로는 1번 사항과 같이 `architecture-level`에서의 최적화를 할 수 있습니다. 
이는, API Gateway와 같은 추가적인 솔루션을 도입하는 개념으로 Load balancing과 같은 큰 범위의 최적화를 의미합니다.
이러한 방식의 최적화는 가장 직관적이고 효과적인 방법 중 하나이지만 비용이 많이 소요된다는 단점이 있습니다.

2번 사항과 같이 `infra-level`에서의 최적화도 고려할 수 있습니다. 
FastAPI 등 인프라를 구성하는 코드에 다양한 최적화 효과를 얻을 수 있습니다. 
본 세미나에서는 이 부분에 대해서 예재와 함께 자세히 알아보고자 합니다.

3번 사항은 Pytorch, Tensorflow 등 실질적으로 ML 서비스가 동작하는 `code-level` 구간에 대한 최적화를 의미합니다.
사실상 ML 서비스에서는 해당 부분이 가장 큰 load가 될 수 밖에 없다는 특징이 있습니다.
이에, 본 세미나에서는 각종 라이브러리를 활용한 최적화부터 시작해서 다양한 방법론을 다루고자 합니다.


### Infra-level 에서의 최적화

#### 1. `pydantic` 라이브러리의 사용을 줄이자! (하이퍼커넥트 게시글 참조)

```python
from pydantic import BaseModel, Field

class Message(BaseModel):
    model: str = Field(..., example="The model name.", description="The model name.")
    options: Optional[dict] = Field(
        {},
        description="A dictionary dictionary for use with the diffusion model.",
        example=_MESSAGE_EXAMPLE_DICT
    )
    request_time: Optional[str] = Field("", description="Log request time")
    note: Optional[str] = Field("", description="Optional notes about this message.")
```

###

현재 Lasco AI에서는 FastAPI를 사용함에 있어 `pydantic`의 `BaseModel`을 통해 API interface를 정의하고 있습니다. 
`pydantic`은 validation check(type check)와 parsing을 매우 편리하게 하는 라이브러리로 대부분의 Fastapi 샘플 코드에서 사용하고 있습니다.

다만, 이러한 `pydantic`은 높은 사용성에 반비례하게 내부 로직이 Pure Python으로 구성되어 있어서 매우 느리다는 단점이 있습니다.
즉, 앞서 설명드린 FastAPI의 강력한 이점을 잃게 되는 것입니다.

이러한 성능차이를 고찰하기 위해 길이가 50인 float list를 가지고 있는 Pydantic 인스턴스 400개를 생성하는 간단한 예시를 살펴보겠습니다.

```python
import timeit
from typing import List

from pydantic import BaseModel

class FeatureSet(BaseModel):
    user_id: int
    features: List[float]

def create_pydantic_instances() -> None:
    for i in range(400):
        obj = FeatureSet(
            user_id=i,
            features=[1.0 * i + j for j in range(50)],
        )

elapsed_time = timeit.timeit(create_pydantic_instances, number=1)
print(f"pydantic: {elapsed_time * 1000:.2f}ms")
```
```shell
pydantic: 12.29ms
```

###

결과를 보면, 위 코드는 단순히 객체만 수백개를 반복적으로 생성했을 뿐인데 12ms라는 시간이 소요되는 것을 확인할 수 있습니다.

그렇다면 `pydantic`이 아닌 바닐라 class로 객체를 생성하면 어떨까요?

```python
import timeit
from typing import List

class FeatureSet:
    def __init__(self, user_id: int, features: List[float]) -> None:
        self.user_id = user_id
        self.features = features
    
def create_class_instances() -> None:
    for i in range(400):
        obj = FeatureSet(
            user_id=i,
            features=[1.0 * i + j for j in range(50)],
        )

elapsed_time = timeit.timeit(create_class_instances, number=1)
print(f"class: {elapsed_time * 1000:.2f}ms")
```
```shell
class: 1.54ms
```

###

코드의 실행 결과는 1.5ms로, `pydantic`을 사용했을 때보다 8배 가량 빠른 것을 확인할 수 있습니다.
물론 해당 차이가 ms 단위로 발생하기 때문에 심각하다고는 할 수는 없으나, 이러한 차이점을 인지하고 있는 것이 중요하다고 생각합니다.

`Pydantic`은 최근 내부 로직을 rust로 작성한 Pydantic V2가 개발 중이며, 라이브러리가 완성되기 전까지는 `pydantic`의 사용을 최소화 하는 것이 좋을 수 있습니다.

###

#### 2. FastAPI의 `def`와 `async def`를 적절하게 사용하자!

FastAPI는 `def`로 함수를 선언하면 자동으로 비동기 처리를 지원하기 때문에 굳이 `async def`를 사용할 필요가 없습니다.
하지만, 그럼에도 불구하고 `async def`가 존재하는 이유는 두 방식에는 분명한 차이가 존재하기 때문인데요, 이 부분에 대해서 자세히 고찰해보도록 하겠습니다.

```python
import time
from fastapi import FastAPI

app = FastAPI()


@app.get("/sync/sync")
def _sync():
    time.sleep(2)
    return {"message": "This is a synchronous route"}


@app.get("/async/async")
async def _async():
    time.sleep(2)
    return {"message": "This is an asynchronous route"}
```

```python
import time
import asyncio
import aiohttp


num_requests = 10


async def send_request(session, url):
    start_time = time.time()
    async with session.get(url) as resp:
        data = await resp.text()
        end_time = time.time()
        print(f"URL: {url}, Status: {resp.status}, Response Time: {end_time - start_time}")


async def main():
    urls = [f"http://localhost:8000/sync/sync" for _ in range(num_requests)] + \
           [f"http://localhost:8000/async/async" for _ in range(num_requests)]
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = send_request(session, url)
            tasks.append(task)
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
```

```shell
URL: http://localhost:8000/async/async, Status: 200, Response Time: 2.0250165462493896
URL: http://localhost:8000/async/async, Status: 200, Response Time: 4.038822174072266
URL: http://localhost:8000/async/async, Status: 200, Response Time: 6.054386138916016
URL: http://localhost:8000/async/async, Status: 200, Response Time: 8.067890644073486
URL: http://localhost:8000/async/async, Status: 200, Response Time: 10.082015991210938
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 10.08301568031311
URL: http://localhost:8000/async/async, Status: 200, Response Time: 12.095595359802246
URL: http://localhost:8000/async/async, Status: 200, Response Time: 14.104830980300903
URL: http://localhost:8000/async/async, Status: 200, Response Time: 16.108787298202515
URL: http://localhost:8000/async/async, Status: 200, Response Time: 18.123507738113403
URL: http://localhost:8000/async/async, Status: 200, Response Time: 20.128472328186035
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 20.130504608154297
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 20.130504608154297
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 20.131505727767944
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 20.131505727767944
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 22.141230821609497
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 22.140230178833008
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 22.140230178833008
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 22.140230178833008
URL: http://localhost:8000/sync/sync, Status: 200, Response Time: 22.141229152679443
```
###

위 실험을 살펴보면 우리가 예상했던 것과는 달리 `def`가 Non-blocking I/O 방식인 것을 볼 수 있고, 
`async def`의 경우 heavy calculation에서 block이 발생하는 것을 볼 수 있습니다.

그렇다면, 우리는 Lasco.AI에서 어떻게 이러한 방법을 사용하고 있는지 살펴보도록 하겠습니다.

###

```python
@router.post("/superlabs_diffusion/v1/img2img", tags=["img2img"], summary="img2img with uploaded image")
async def img2img_api(message_json: str = Form(...), file: UploadFile = File(...), diffusion_model=Depends(get_diffusion_model)) -> Response:
    pass
```

###

Lasco AI에서는 대부분 `async def` 방식을 사용하고 있는데, 이는 마치 queue를 통한 순차 처리의 방식과 유사하게 적용되고 있는 것을 알 수 있습니다.

그렇다면, 왜 동시/병렬적 처리가 가능한 `def`를 사용하지 않고 `async def`를 사용하고 있을까요?
이는 GPU Memory의 Overhead와 큰 관련이 있습니다.

AI 서비스는 그 특성상 많은 GPU Memory를 점유한채로 GPU 연산을 수행해야 합니다. 
따라서, 동시/병렬적인 요청이 발생하면 VRAM에서 OOM(Out Of Memory)가 발생하는 등 오류가 발생할 가능성이 높아지게 됩니다.

실제로 Diffusion 모델에서도 `def` 방식을 통해 동시/병렬적으로 여러 요청을 보내게 되면, 이미지를 재대로 생성하지 못하는 문제가 있습니다.

이러한 특징에 따라 AI 서비스를 구성함에 있어, Computing power나 API의 목적 등을 고려하여 `def`와 `async def` 중 적절한 방법을 사용하는 것이 중요합니다.

###

### code-level 에서의 최적화

#### 1. `numba` 라이브러리를 사용하여 최적화 해보자!

`numba`를 통해 numpy array, function 등을 이용한 Python 코드를 머신코드로 JIT 컴파일을 수행하는 것도 좋은 최적화의 방법이 될 수 있습니다.
`numba`는 python 코드의 일부를 컴파일하여 C, C++에 준하는 속도를 달성하게 도와줍니다.
다음은 이러한 최적화에 대한 이해를 돕기 위한 간단한 예시입니다.

```python
from numba import jit
import timeit
import time


def calculate_factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result


@jit(nopython=True, cache=True)
def calculate_factorial_with_numba(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result


# pure Python에 대한 측정
start_time_python = time.time()
factorial_python = calculate_factorial(100000)
end_time_python = time.time()
elapsed_time_python = end_time_python - start_time_python
print(f"Elapsed Time (pure python) = {elapsed_time_python}")

# Numba 적용 후에 대한 측정
calculate_factorial_with_numba(1)
numba_time = timeit.timeit(
    "calculate_factorial_with_numba(100000)",
    setup="from __main__ import calculate_factorial_with_numba",
    number=100
)

average_numba_time = numba_time / 100
print(f"Average time (numba): {average_numba_time} seconds")

# numba_time이 elapsed_time_python에 비해 몇 배 작은지 계산
speedup = elapsed_time_python / average_numba_time
print(f"Numba is {speedup} times faster than pure Python.")
```

```shell
Elapsed Time (pure python) = 1.8644723892211914
Average time (numba): 2.537000000302214e-06 seconds
Numba is 734912.2542369297 times faster than pure Python.
```

###

예재 코드의 결과를 살펴보면 매우 큰 속도 차이를 볼 수 있습니다. 
`numba`로 최적화한 코드는 같은 작업을 수행하는 pure python 코드에 비해 약 734,912배 더 빠르게 실행되었습니다.

하지만 이번 케이스의 결과처럼 `numba`가 모든 작업에서 이렇게 높은 성능 향상을 가져다 주는 것은 아닙니다.
팩토리얼 계산과 같은 단순 반복 작업에서는 Numba 최적화가 큰 효과를 보여주고 있지만, 다른 task에서는 이보다 적은 향상이나 전혀 향상되지 않을 수도 있습니다.

따라서, 성능 최적화의 문제는 항상 task의 문제의 맥락을 고려하고 적합성 여부를 판단해서 진행하는 것이 좋다는 점을 인지하고 있어야 합니다.

기회가 된다면, Lasco AI에서 이러한 `numba`가 적용될 수 있는 코드를 찾아 모듈화 하여 최적화를 시도해보면 좋을 것 같습니다.

###

#### 2. `diffusers` 라이브러리를 최적화 해보자!

`Diffusers`에서 공식적으로 지원하는 최적화 기능을 적용하는 것도 code-level 최적화의 방법 중 하나가 될 수 있습니다.
특히, `diffusers` 커뮤니티에서는 최적화에 대한 업데이트가 활발하게 이루어지고 있으므로, 이 부분을 잘 follow-up 하는 것도 많은 정보를 얻을 수 있습니다.

다음은 `diffusers`에서 공식적으로 검증이 완료된 몇 가지 최적화의 예재를 보여드리고자 합니다.

###

##### Sliced attention

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_attention_slicing()
image = pipe(prompt).images[0]
```
###

`Sliced attention`은 메모리 절약을 위해 attention 연산을 한번에 모두 계산하는 대신 단계적으로 수행하게 만드는 최적화 기법입니다.
이를 통해 inference time이 약 10% 정도 느려지지만, VRAM의 사용량이 3.2GB로 감소한다는 연구 결과가 있습니다.

###

##### VAE tiling

```python
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
prompt = "a beautiful landscape photograph"
pipe.enable_vae_tiling()
pipe.enable_xformers_memory_efficient_attention()

image = pipe([prompt], width=3840, height=2224, num_inference_steps=20).images[0]
```
###

`VAE tiling`을 통해 큰 사이즈의 이미지를 생성하는 과정에서 메모리를 절약할 수 있습니다.
이 부분은 이미 Lasco.AI에서 사용해봤던 기술이기 때문에 자세한 설명은 생략하도록 하겠습니다.

이러한 최적화 기술의 목적은 일반적으로 Super resolution과 같은 업스케일 task에서 입력 이미지의 크기는 computing power와 연관성이 높은 것으로 알려져 있는데,
메모리 총 사용량을 줄이기 위한 요소로 사용되고 있습니다.

###

##### Model offloading

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  
    torch_dtype=torch.float16,
)

prompt = "a photo of an astronaut riding a horse on mars"
pipe.enable_model_cpu_offload()
image = pipe(prompt).images[0]
```

###

`Model offloading` 기술을 적용하여 CPU와 GPU 간의 전환을 최소화하여 최적화 할 수 있습니다.
CPU offloading은 폰 노이만 구조에서는 반드시 효과가 있을 수밖에 없는 요소로 `diffusers`에서도 최근 이러한 offloading을 지원하게 되었습니다.
`enable_model_cpu_offload()` 함수를 사용하여 이러한 최적화 기술을 쉽게 적용해 볼 수 있습니다.

###

##### Torch compile

```python
from diffusers import DiffusionPipeline
import torch

path = "runwayml/stable-diffusion-v1-5"


pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "ghibli style, a fantasy landscape with castles"
```

###

`torch.compile`을 통해 Pytorch 코드로 구성된 모델을 JIT 컴파일하여 실행속도를 최적화할 수 있습니다.
`diffusers`에서도 이러한 `torch.compile`과 호환이 되기 때문에 이러한 최적화 기술을 쉽게 적용해 볼 수 있습니다.

현재 이러한 compile 기술들이 적용되어 있지 않은데, 기회가 된다면 실제 Lasco.AI에 적용하여 전/후를 비교해 보는 것도 좋을 것 같습니다.

###

## 맺으며

본 세미나에서는 Python을 기반으로 구성되어 있는 라스코의 AI 백엔드 서버를 최적화하기 위한 다양한 방법을 고찰하였습니다.

경량화/최적화의 분야는 굉장히 다양한 방면에서 연구되고 있는 주제인 만큼, 
구축하고자 하는 서비스에 대해서 충분히 고민하고, 그에 맞는 기술을 꾸준히 follow-up 하는 것이 중요할 것 같습니다.