@php
use App\Models\Progress;

@endphp

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ asset('css/dashboard.css') }}" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
    <title>Dashboard</title>
</head>

<body>


    @include('navbar');
    <!-- content -->
    <div class="container">
        <div class="container-wrapper">
            <div class="content1">
                <div class="profile-content-2">
                    <div class="header">
                        <div class="text">Welcome,</div>
                        <h2>{{ $user->full_name }}</h2>
                    </div>
                    <div class="progress">
                        <div class="task">
                            <h3>100</h3>
                            <p>Tasks Completed</p>
                        </div>
                        <div class="lesson">
                            <h3>23</h3>
                            <p>Lessons Completed</p>
                        </div>
                    </div>
                </div>
                <img src="{{ asset('elements/dash_image2.png') }}" alt="">
            </div>
            <div class="content2">
                <div class="lesson-content">
                    <h1>Lessons</h1>
                    <div class="lesson-cards">
                        @foreach($lessons as $lesson)
                        <div class="card">
                            <div class="card-2">
                                <div class="card-content">
                                    <h3>{{ $lesson->lesson_name }}</h3>
                                    <p>{{ $lesson->lesson_desc }}</p>

                                    @php
                                    $lessonProgress = Progress::where('lesson_id', $lesson->lesson_id)
                                    ->where('user_id', $user->user_id)
                                    ->first();
                                    @endphp

                                    <div class="status">
                                        <p>Status:</p>
                                        <div class="text">{{ optional($lessonProgress)->is_completed ? 'Completed' : 'Not Completed' }}</div>
                                    </div>

                                    <div class="bawah">
                                        <form action="{{ route('task.page', ['lesson_id' => $lesson->lesson_id]) }}" method="GET" style="display:inline;">
                                            <button type="submit" class="signup">Resume<span class="material-symbols-outlined">
                                                    trending_flat
                                                </span></button>
                                        </form>
                                    </div>
                                </div>
                            </div>
                        </div>
                        @endforeach
                        <div class="cardtwo">
                            <div class="card-2">
                                <div class="card-content">
                                    <h3><span class="material-symbols-outlined">
                                            lock
                                        </span> Numbers in Sign Language</h3>
                                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                                    <div class="bawah">

                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="cardthree">
                            <div class="card-2">
                                <div class="card-content">
                                    <h3><span class="material-symbols-outlined">
                                            lock
                                        </span> Everyday Phrases 101</h3>
                                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
                                    <div class="bawah">

                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

</html>