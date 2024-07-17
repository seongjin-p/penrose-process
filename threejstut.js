import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// 씬 생성
const scene = new THREE.Scene();

// 카메라 생성
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 100000);
camera.position.set(0, 30000, 50000); // 카메라 위치 조정
camera.lookAt(0, 0, 0); // 가운데를 향하도록 설정

// 렌더러 생성
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// OrbitControls 추가
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; // 부드러운 카메라 이동
controls.dampingFactor = 0.25;
controls.screenSpacePanning = false;
controls.maxPolarAngle = Math.PI / 2;
controls.enableZoom = true; // 확대/축소 가능하게 설정

// AmbientLight 추가
const ambientLight = new THREE.AmbientLight(0x404040, 2); // Ambient light 추가
scene.add(ambientLight);

// DirectionalLight 추가
const directionalLight = new THREE.DirectionalLight(0xFFFFFF, 1);
directionalLight.position.set(0, 60000, 0); // 조명 위치 설정
directionalLight.target.position.set(0, 0, 0); // 조명이 바라보는 위치 설정
directionalLight.castShadow = true;
directionalLight.shadow.camera.left = -100000;
directionalLight.shadow.camera.right = 100000;
directionalLight.shadow.camera.top = 100000;
directionalLight.shadow.camera.bottom = -100000;
directionalLight.shadow.camera.near = 0.5;
directionalLight.shadow.camera.far = 200000;
scene.add(directionalLight);
scene.add(directionalLight.target);

// SpotLight 추가
const spotLight = new THREE.SpotLight(0xFFFFFF);
spotLight.position.set(100000, 100000, 100000);
spotLight.angle = Math.PI / 4;
spotLight.penumbra = 0.1;
spotLight.decay = 2;
spotLight.distance = 500000;
spotLight.castShadow = true;
scene.add(spotLight);

// PointLight 추가
const pointLight = new THREE.PointLight(0xFFFFFF, 1, 500000);
pointLight.position.set(0, 30000, 0);
scene.add(pointLight);


const loader = new GLTFLoader(); // 3D 모델 로더

let blackhole; // 블랙홀 변수
let spaceship; // 우주선 변수

// 블랙홀 로더
loader.load(
    '3D_models/blackhole/blackhole.gltf',
    function (gltf) {
        blackhole = gltf.scene;

        // 블랙홀 크기 조정
        const box = new THREE.Box3().setFromObject(blackhole);
        const size = box.getSize(new THREE.Vector3());
        const desiredSize = new THREE.Vector3(9000, 3000, 9000);
        const scaleFactor = new THREE.Vector3(
            desiredSize.x / size.x,
            desiredSize.y / size.y,
            desiredSize.z / size.z
        );
        blackhole.scale.set(scaleFactor.x, scaleFactor.y, scaleFactor.z);

        scene.add(blackhole); // scene에 추가

        console.log('Blackhole loaded and added to scene');
    },
    undefined,
    function (error) {
        console.error('An error occurred while loading the blackhole:', error);
    }
);

// 우주선 로더
loader.load(
    '3D_models/spaceship/spaceship.gltf',
    function (gltf) {
        spaceship = gltf.scene;
        spaceship.scale.set(100, 100, 100); // 크기를 조정
        spaceship.position.set(4012, 0, 0); // 초기 위치 설정

        scene.add(spaceship); // scene에 추가

        console.log('Spaceship loaded and added to scene');
    },
    undefined,
    function (error) {
        console.error('An error occurred while loading the spaceship:', error);
    }
);

// 우주선 이동 함수
function moveSpaceship(t) {
    if (spaceship) {
        const x = -0.0000 + (0.1159 * Math.pow(t, 2)) + (-79.0854 * t) + 5224.9392;
        const z = 0.0001 + (-0.1240 * Math.pow(t, 2)) + (2.6246 * t) + 4845.7978;
        spaceship.position.set(x, spaceship.position.y, z);
        console.log(`Spaceship position: x=${x}, z=${z}`); // 위치를 확인하기 위한 로그
    }
}

// 애니메이션 함수
const animate = function () {
    requestAnimationFrame(animate);

    // 현재 시간 (초 단위로 변환)
    const t = (Date.now() % 10000) / 10; // 0에서 10초 사이의 값 반복

    // 우주선 이동
    moveSpaceship(t);

    // 블랙홀 회전
    if (blackhole) {
        blackhole.rotation.y += 0.015;
    }

    // 렌더링
    renderer.render(scene, camera);
};

// 창 크기 변경 시 카메라와 렌더러 업데이트
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();
