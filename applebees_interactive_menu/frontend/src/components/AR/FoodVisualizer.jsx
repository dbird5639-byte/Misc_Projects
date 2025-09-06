import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { ARButton } from 'three/examples/jsm/webxr/ARButton';
import { VRButton } from 'three/examples/jsm/webxr/VRButton';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useSpring, animated } from '@react-spring/three';
import { useFrame } from '@react-three/fiber';
import { Canvas, useLoader, useThree } from '@react-three/fiber';
import { Environment, PresentationControls, Float, Text } from '@react-three/drei';

import './FoodVisualizer.css';

const FoodVisualizer = ({ 
  itemId, 
  itemName, 
  modelUrl, 
  textureUrl, 
  onInteraction,
  isARMode = false 
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [currentAnimation, setCurrentAnimation] = useState('idle');
  const [interactionMode, setInteractionMode] = useState('view');
  const [arSession, setArSession] = useState(null);
  const [performanceMetrics, setPerformanceMetrics] = useState({
    fps: 60,
    renderTime: 0,
    memoryUsage: 0
  });

  const sceneRef = useRef();
  const rendererRef = useRef();
  const mixerRef = useRef();
  const clockRef = useRef(new THREE.Clock());

  // Animation states
  const [animations, setAnimations] = useState({
    idle: true,
    sauce_drip: false,
    steam: false,
    rotate: false
  });

  // Interactive elements
  const [interactiveElements, setInteractiveElements] = useState({
    sauce_bottle: { visible: false, position: [0.1, 0.05, 0] },
    ingredients: { visible: false, position: [-0.1, 0.05, 0] }
  });

  // Load 3D model
  const gltf = useLoader(GLTFLoader, modelUrl);
  const texture = useLoader(THREE.TextureLoader, textureUrl);

  useEffect(() => {
    if (gltf) {
      setIsLoaded(true);
      setupModel(gltf);
      setupAnimations(gltf);
      setupLighting();
      setupInteractions();
    }
  }, [gltf]);

  const setupModel = useCallback((gltf) => {
    const model = gltf.scene;
    
    // Apply texture
    model.traverse((child) => {
      if (child.isMesh) {
        child.material.map = texture;
        child.material.needsUpdate = true;
        
        // Enable shadows
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });

    // Scale and position model
    model.scale.set(1, 1, 1);
    model.position.set(0, 0, 0);

    sceneRef.current.add(model);
  }, [texture]);

  const setupAnimations = useCallback((gltf) => {
    if (gltf.animations && gltf.animations.length > 0) {
      mixerRef.current = new THREE.AnimationMixer(gltf.scene);
      
      gltf.animations.forEach((clip) => {
        const action = mixerRef.current.clipAction(clip);
        if (clip.name === 'idle') {
          action.play();
        }
      });
    }
  }, []);

  const setupLighting = useCallback(() => {
    // Main light
    const mainLight = new THREE.DirectionalLight(0xffffff, 1);
    mainLight.position.set(0, 2, 2);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    sceneRef.current.add(mainLight);

    // Fill light
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
    fillLight.position.set(-1, 1, 1);
    sceneRef.current.add(fillLight);

    // Ambient light
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    sceneRef.current.add(ambientLight);

    // Rim light
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.2);
    rimLight.position.set(1, 1, -1);
    sceneRef.current.add(rimLight);
  }, []);

  const setupInteractions = useCallback(() => {
    // Add interactive elements
    const sauceBottle = createInteractiveElement('sauce_bottle', [0.1, 0.05, 0]);
    const ingredients = createInteractiveElement('ingredients', [-0.1, 0.05, 0]);
    
    sceneRef.current.add(sauceBottle);
    sceneRef.current.add(ingredients);
  }, []);

  const createInteractiveElement = (name, position) => {
    const geometry = new THREE.SphereGeometry(0.02, 16, 16);
    const material = new THREE.MeshBasicMaterial({ 
      color: 0x00ff00, 
      transparent: true, 
      opacity: 0.7 
    });
    
    const element = new THREE.Mesh(geometry, material);
    element.position.set(...position);
    element.name = name;
    element.userData = { interactive: true, type: name };
    
    return element;
  };

  // Animation loop
  useFrame((state, delta) => {
    if (mixerRef.current) {
      mixerRef.current.update(delta);
    }

    // Update performance metrics
    const renderTime = state.clock.elapsedTime;
    setPerformanceMetrics(prev => ({
      ...prev,
      renderTime: renderTime,
      fps: 1 / delta
    }));

    // Handle animations
    handleAnimations(delta);
  });

  const handleAnimations = useCallback((delta) => {
    if (animations.sauce_drip) {
      // Animate sauce dripping
      const sauceEffect = sceneRef.current.getObjectByName('sauce_effect');
      if (sauceEffect) {
        sauceEffect.position.y -= delta * 0.1;
        if (sauceEffect.position.y < -0.5) {
          setAnimations(prev => ({ ...prev, sauce_drip: false }));
        }
      }
    }

    if (animations.steam) {
      // Animate steam effect
      const steamEffect = sceneRef.current.getObjectByName('steam_effect');
      if (steamEffect) {
        steamEffect.position.y += delta * 0.2;
        steamEffect.material.opacity -= delta * 0.5;
        if (steamEffect.material.opacity <= 0) {
          setAnimations(prev => ({ ...prev, steam: false }));
        }
      }
    }

    if (animations.rotate) {
      // Rotate model
      const model = sceneRef.current.children.find(child => child.type === 'Group');
      if (model) {
        model.rotation.y += delta * 0.5;
      }
    }
  }, [animations]);

  // AR Setup
  useEffect(() => {
    if (isARMode && rendererRef.current) {
      setupAR();
    }
  }, [isARMode]);

  const setupAR = useCallback(async () => {
    try {
      // Check WebXR support
      if (!navigator.xr) {
        console.warn('WebXR not supported');
        return;
      }

      // Check AR support
      const isARSupported = await navigator.xr.isSessionSupported('immersive-ar');
      if (!isARSupported) {
        console.warn('AR not supported');
        return;
      }

      // Setup AR session
      const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['hit-test', 'dom-overlay'],
        domOverlay: { root: document.getElementById('ar-overlay') }
      });

      setArSession(session);

      // Setup AR rendering
      rendererRef.current.xr.setReferenceSpaceType('local');
      rendererRef.current.xr.setSession(session);

      // Add AR button
      const arButton = ARButton.createButton(rendererRef.current, {
        sessionInit: {
          requiredFeatures: ['hit-test', 'dom-overlay'],
          domOverlay: { root: document.getElementById('ar-overlay') }
        }
      });

      document.body.appendChild(arButton);

      // Handle AR session end
      session.addEventListener('end', () => {
        setArSession(null);
        document.body.removeChild(arButton);
      });

    } catch (error) {
      console.error('AR setup failed:', error);
    }
  }, []);

  // Interaction handlers
  const handleTap = useCallback((event) => {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, rendererRef.current.camera);

    const intersects = raycaster.intersectObjects(sceneRef.current.children, true);

    if (intersects.length > 0) {
      const object = intersects[0].object;
      
      if (object.userData.interactive) {
        handleInteractiveElement(object.userData.type);
      }
    }
  }, []);

  const handleInteractiveElement = useCallback((elementType) => {
    switch (elementType) {
      case 'sauce_bottle':
        setAnimations(prev => ({ ...prev, sauce_drip: true }));
        onInteraction?.('sauce_bottle', { action: 'show_sauce_options' });
        break;
      
      case 'ingredients':
        setAnimations(prev => ({ ...prev, steam: true }));
        onInteraction?.('ingredients', { action: 'show_ingredients' });
        break;
      
      default:
        break;
    }
  }, [onInteraction]);

  const handleSwipe = useCallback((direction) => {
    setAnimations(prev => ({ ...prev, rotate: true }));
    
    setTimeout(() => {
      setAnimations(prev => ({ ...prev, rotate: false }));
    }, 1000);
  }, []);

  const handleVoiceCommand = useCallback((command) => {
    const commandLower = command.toLowerCase();
    
    if (commandLower.includes('rotate')) {
      handleSwipe('right');
    } else if (commandLower.includes('sauce')) {
      handleInteractiveElement('sauce_bottle');
    } else if (commandLower.includes('ingredients')) {
      handleInteractiveElement('ingredients');
    }
  }, [handleSwipe, handleInteractiveElement]);

  // Performance monitoring
  useEffect(() => {
    const interval = setInterval(() => {
      if (performance.memory) {
        setPerformanceMetrics(prev => ({
          ...prev,
          memoryUsage: performance.memory.usedJSHeapSize / 1024 / 1024 // MB
        }));
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  if (!isLoaded) {
    return (
      <div className="food-visualizer-loading">
        <div className="loading-spinner"></div>
        <p>Loading 3D model...</p>
      </div>
    );
  }

  return (
    <div className="food-visualizer-container">
      {/* 3D Scene */}
      <Canvas
        ref={rendererRef}
        camera={{ position: [0, 0, 5], fov: 75 }}
        shadows
        onCreated={({ gl, scene }) => {
          gl.shadowMap.enabled = true;
          gl.shadowMap.type = THREE.PCFSoftShadowMap;
          sceneRef.current = scene;
        }}
        onClick={handleTap}
      >
        {/* Environment */}
        <Environment preset="restaurant" />
        
        {/* Food Model */}
        <Float
          speed={1.5}
          rotationIntensity={0.5}
          floatIntensity={0.5}
        >
          <primitive object={gltf.scene} />
        </Float>

        {/* Interactive Elements */}
        {Object.entries(interactiveElements).map(([name, element]) => (
          <mesh
            key={name}
            position={element.position}
            visible={element.visible}
            userData={{ interactive: true, type: name }}
          >
            <sphereGeometry args={[0.02, 16, 16]} />
            <meshBasicMaterial color={0x00ff00} transparent opacity={0.7} />
          </mesh>
        ))}

        {/* Effects */}
        {animations.sauce_drip && (
          <mesh name="sauce_effect" position={[0, 0.1, 0]}>
            <cylinderGeometry args={[0.01, 0.01, 0.5]} />
            <meshBasicMaterial color={0xff6600} transparent opacity={0.8} />
          </mesh>
        )}

        {animations.steam && (
          <mesh name="steam_effect" position={[0, 0.2, 0]}>
            <sphereGeometry args={[0.05, 16, 16]} />
            <meshBasicMaterial color={0xcccccc} transparent opacity={0.6} />
          </mesh>
        )}

        {/* Controls */}
        {!isARMode && (
          <PresentationControls
            global
            config={{ mass: 2, tension: 500 }}
            snap={{ mass: 4, tension: 1500 }}
            rotation={[0, 0, 0]}
            polar={[-Math.PI / 3, Math.PI / 3]}
            azimuth={[-Math.PI / 1.4, Math.PI / 2]}>
            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
            />
          </PresentationControls>
        )}
      </Canvas>

      {/* AR Overlay */}
      {isARMode && (
        <div id="ar-overlay" className="ar-overlay">
          <div className="ar-controls">
            <button onClick={() => handleInteractiveElement('sauce_bottle')}>
              Show Sauces
            </button>
            <button onClick={() => handleInteractiveElement('ingredients')}>
              Show Ingredients
            </button>
            <button onClick={() => handleSwipe('right')}>
              Rotate
            </button>
          </div>
        </div>
      )}

      {/* Controls Panel */}
      <div className="controls-panel">
        <h3>{itemName}</h3>
        
        <div className="control-buttons">
          <button 
            onClick={() => handleInteractiveElement('sauce_bottle')}
            className="control-btn sauce-btn"
          >
            🍯 Sauces
          </button>
          
          <button 
            onClick={() => handleInteractiveElement('ingredients')}
            className="control-btn ingredients-btn"
          >
            🥬 Ingredients
          </button>
          
          <button 
            onClick={() => handleSwipe('right')}
            className="control-btn rotate-btn"
          >
            🔄 Rotate
          </button>
          
          <button 
            onClick={() => setAnimations(prev => ({ ...prev, steam: true }))}
            className="control-btn steam-btn"
          >
            💨 Steam Effect
          </button>
        </div>

        {/* Performance Metrics */}
        <div className="performance-metrics">
          <div className="metric">
            <span>FPS:</span>
            <span>{Math.round(performanceMetrics.fps)}</span>
          </div>
          <div className="metric">
            <span>Memory:</span>
            <span>{performanceMetrics.memoryUsage.toFixed(1)} MB</span>
          </div>
        </div>
      </div>

      {/* Voice Command Interface */}
      <div className="voice-interface">
        <button 
          className="voice-btn"
          onClick={() => {
            if ('webkitSpeechRecognition' in window) {
              const recognition = new webkitSpeechRecognition();
              recognition.continuous = false;
              recognition.interimResults = false;
              
              recognition.onresult = (event) => {
                const command = event.results[0][0].transcript;
                handleVoiceCommand(command);
              };
              
              recognition.start();
            }
          }}
        >
          🎤 Voice Command
        </button>
      </div>
    </div>
  );
};

export default FoodVisualizer; 