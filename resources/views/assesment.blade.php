<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ asset('css/task.css') }}" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
    <title>Tutur</title>
</head>

<body>
    <!-- Navbar -->
    <div class="navbar">
        <div class="nav-wrapper">
            <a href="{{ route('dashboard.page') }}"><span class="material-symbols-outlined">
                    keyboard_backspace
                </span>Dashboard</a>
            <h3>Sign Language ABCs</h3>
        </div>
    </div>
    <!-- Content -->
    <div class="container">
        <div class="content-container">
            <div class="col-1">
                @foreach ($tasks as $task)
                <div class="box">
                    <div class="box-container">
                        <a href="{{ route('task.page2', ['lesson_id' => $task->lesson_id, 'task_id' => $task->task_id]) }}" class="task-link" data-task-id="{{ $task->task_id }}">{{ $task->task_name }}</a>
                    </div>
                </div>
                @endforeach
                <div class="box">
                    <div class="box-container">
                        <a href="#" class="task-link" data-task-id="assessment">Assessment</a>
                    </div>
                </div>
            </div>
            <div class="col-2">
                <div class="camera">
                    <div class="top">
                        <h4>Assessment</h4>
                        <button onclick="toggleCamera()">Camera <span class="material-symbols-outlined">
                                videocam
                            </span></button>
                    </div>
                    <p>Sign out the letters you just learned and close the camera afterwards to record the progress!</p>
                    <div class="cam">
                    <img src="http://localhost:5000/video_feed_assessment" id="video" style="display: none;" alt="Live Video Stream">
                    </div>
                    <form method="POST" action="{{ route('completeProgress', ['lesson_id' => $lesson_id, 'progress_id' => $progress_id]) }}">
                        @csrf
                        <button type="submit" class="complete">Complete <span class="material-symbols-outlined">
                                trending_flat
                            </span></button>
                    </form>
                </div>
            </div>
        </div>
    </div>
    <script>
        function toggleCamera() {
            var video = document.getElementById("video");
            if (video.style.display === "none") {
                video.style.display = "block";
            } else {
                video.style.display = "none";
            }
        }

        document.addEventListener('DOMContentLoaded', function() {
            const taskLinks = document.querySelectorAll('.task-link');
            const boxContainers = document.querySelectorAll('.box-container');

            // Initialize all .box-container elements with radio_button_unchecked
            boxContainers.forEach(container => {
                const span = document.createElement('span');
                span.className = 'material-symbols-outlined';
                span.innerText = 'radio_button_unchecked';
                container.appendChild(span);
            });

            // Check localStorage for previously selected task
            const selectedTaskId = localStorage.getItem('selectedTaskId');
            if (selectedTaskId) {
                const selectedLink = document.querySelector(`a[data-task-id="${selectedTaskId}"]`);
                if (selectedLink) {
                    const span = selectedLink.parentElement.querySelector('span');
                    if (span) {
                        span.innerText = 'radio_button_checked';
                    }
                }
            }

            taskLinks.forEach(link => {
                link.addEventListener('click', function(event) {
                    // Store the task ID in localStorage
                    const taskId = event.target.getAttribute('data-task-id');
                    localStorage.setItem('selectedTaskId', taskId);

                    // Remove existing span and add unchecked span to all .box-container elements
                    boxContainers.forEach(container => {
                        const span = container.querySelector('span');
                        if (span) {
                            span.innerText = 'radio_button_unchecked';
                        }
                    });

                    // Change the span to checked for the clicked link's parent container
                    const span = event.target.parentElement.querySelector('span');
                    if (span) {
                        span.innerText = 'radio_button_checked';
                    }

                    // Update button data-task-id attributes
                    document.querySelector('.back').setAttribute('data-task-id', taskId);
                    document.querySelector('.next').setAttribute('data-task-id', taskId);
                });
            });
        });

        function navigateTask(direction) {
            const taskLinks = Array.from(document.querySelectorAll('.task-link'));
            const currentTaskId = parseInt(localStorage.getItem('selectedTaskId'));
            let currentIndex = taskLinks.findIndex(link => parseInt(link.getAttribute('data-task-id')) === currentTaskId);

            if (direction === 'previous' && currentIndex > 0) {
                currentIndex -= 1;
            } else if (direction === 'next' && currentIndex < taskLinks.length - 1) {
                currentIndex += 1;
            } else {
                return; // Do nothing if at the bounds
            }

            const newTaskId = taskLinks[currentIndex].getAttribute('data-task-id');
            localStorage.setItem('selectedTaskId', newTaskId);
            window.location.href = taskLinks[currentIndex].getAttribute('href');
        }
    </script>
</body>

</html>