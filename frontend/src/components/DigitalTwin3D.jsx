import React, { useEffect, useRef } from "react";
import * as THREE from "three";

const STATE_COLORS = {
  HEALTHY: 0x38f5c8,
  SUSPECT: 0xf5a623,
  DOWN: 0xf5456c,
};

/**
 * Renders the mesh topology as an interactive 3D digital twin: each node
 * is a glowing sphere positioned on a ring, edges are lit lines, and node
 * color/pulse reflects live health state from the telemetry snapshot.
 */
export default function DigitalTwin3D({ topology, className = "" }) {
  const mountRef = useRef(null);
  const sceneState = useRef(null);

  // -- one-time Three.js scene setup ------------------------------------
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth;
    const height = mount.clientHeight;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0a0e14, 0.035);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(0, 4.5, 9);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0x8994a8, 0.6);
    scene.add(ambient);
    const point = new THREE.PointLight(0x7c6cf6, 2.2, 30);
    point.position.set(0, 6, 4);
    scene.add(point);

    // Ground grid, subtle
    const grid = new THREE.GridHelper(20, 24, 0x1e2634, 0x161c27);
    grid.position.y = -1.6;
    scene.add(grid);

    // Central "self" node
    const coreGeom = new THREE.IcosahedronGeometry(0.55, 1);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0x7c6cf6,
      emissive: 0x2a2170,
      metalness: 0.4,
      roughness: 0.25,
    });
    const core = new THREE.Mesh(coreGeom, coreMat);
    scene.add(core);

    const nodeGroup = new THREE.Group();
    scene.add(nodeGroup);
    const edgeGroup = new THREE.Group();
    scene.add(edgeGroup);

    let raf = null;
    const clock = new THREE.Clock();

    function animate() {
      const t = clock.getElapsedTime();
      core.rotation.y = t * 0.25;
      core.rotation.x = Math.sin(t * 0.2) * 0.15;
      nodeGroup.children.forEach((mesh, i) => {
        mesh.rotation.y = t * 0.4 + i;
        const pulse = mesh.userData.pulse || 0;
        if (pulse > 0) {
          const s = 1 + Math.sin(t * 10) * 0.12 * pulse;
          mesh.scale.setScalar(s);
        }
      });
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    }
    animate();

    function handleResize() {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(mount);

    sceneState.current = { scene, camera, renderer, nodeGroup, edgeGroup };

    return () => {
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

  // -- rebuild node/edge meshes whenever topology updates ---------------
  useEffect(() => {
    const state = sceneState.current;
    if (!state || !topology) return;
    const { nodeGroup, edgeGroup } = state;

    // Clear previous frame's meshes
    [...nodeGroup.children].forEach((c) => nodeGroup.remove(c));
    [...edgeGroup.children].forEach((c) => edgeGroup.remove(c));

    const nodes = topology.nodes || [];
    const radius = 3.4;
    const positions = {};

    nodes.forEach((node, i) => {
      const angle = (i / Math.max(nodes.length, 1)) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = Math.sin(i * 1.7) * 0.4;
      positions[node.node_id] = new THREE.Vector3(x, y, z);

      const color = STATE_COLORS[node.state] ?? 0x8994a8;
      const geom = new THREE.SphereGeometry(0.24, 24, 24);
      const mat = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: node.state === "DOWN" ? 0.15 : 0.45,
        metalness: 0.3,
        roughness: 0.35,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.copy(positions[node.node_id]);
      mesh.userData.pulse = node.state === "SUSPECT" ? 1 : node.state === "DOWN" ? 0.4 : 0;
      nodeGroup.add(mesh);

      // Ring around DOWN nodes to make failures unmistakable
      if (node.state === "DOWN") {
        const ringGeom = new THREE.RingGeometry(0.34, 0.4, 32);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0xf5456c,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.7,
        });
        const ring = new THREE.Mesh(ringGeom, ringMat);
        ring.position.copy(positions[node.node_id]);
        ring.lookAt(0, ring.position.y, 0);
        nodeGroup.add(ring);
      }
    });

    // Edge lines: self (origin) -> each node, plus peer-to-peer edges
    const edges = topology.edges || [];
    edges.forEach((edge) => {
      const a = edge.a === (topology.selfId ?? "self") ? new THREE.Vector3(0, 0, 0) : positions[edge.a];
      const b = edge.b === (topology.selfId ?? "self") ? new THREE.Vector3(0, 0, 0) : positions[edge.b];
      if (!a || !b) return;
      const isDownLink =
        nodes.find((n) => n.node_id === edge.a)?.state === "DOWN" ||
        nodes.find((n) => n.node_id === edge.b)?.state === "DOWN";
      const geom = new THREE.BufferGeometry().setFromPoints([a, b]);
      const mat = new THREE.LineBasicMaterial({
        color: isDownLink ? 0x3a2530 : 0x38f5c8,
        transparent: true,
        opacity: isDownLink ? 0.15 : 0.45,
      });
      edgeGroup.add(new THREE.Line(geom, mat));
    });

    // Fallback for nodes not present in `edges` (e.g. simulated data
    // without an explicit self-node id) — connect every node to origin.
    if (edges.length === 0) {
      nodes.forEach((node) => {
        const pos = positions[node.node_id];
        const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), pos]);
        const mat = new THREE.LineBasicMaterial({
          color: node.state === "DOWN" ? 0x3a2530 : 0x38f5c8,
          transparent: true,
          opacity: node.state === "DOWN" ? 0.15 : 0.45,
        });
        edgeGroup.add(new THREE.Line(geom, mat));
      });
    }
  }, [topology]);

  return <div ref={mountRef} className={`w-full h-full ${className}`} />;
}
