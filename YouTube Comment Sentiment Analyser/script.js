
 // Set up Three.js scene
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Load YouTube logo model
    const loader = new THREE.GLTFLoader();
    loader.load('https://threejs.org/examples/models/gltf/YouTube.glb', function (gltf) {
        scene.add(gltf.scene);
    }, undefined, function (error) {
        console.error(error);
    });

    // Position the camera
    camera.position.z = 100;

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        // Render the scene
        renderer.render(scene, camera);
    }
    animate();

    // Add event listener to the Analyze Comments button
    document.querySelector('.button').addEventListener('click', function() {
        var youtubeLink = document.getElementById('youtubeLink').value;
        fetch('/analyze', {
            method: 'POST',
            body: JSON.stringify({youtubeLink: youtubeLink}),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            var likesCount = data.likesCount;
            var likesBar = document.getElementById('likesBar');
            likesBar.style.width = likesCount + 'px';
            document.getElementById('commentAnalysis').innerHTML = '<p>Sentiment Analysis:</p><ul>';
            data.sentimentLabels.forEach(label => {
                document.getElementById('commentAnalysis').innerHTML += '<li>' + label + '</li>';
            });
            document.getElementById('commentAnalysis').innerHTML += '</ul>';
            console.log('Accuracy:', data.accuracy);
        })
        .catch(error => console.error('Error:', error));
    });

    
