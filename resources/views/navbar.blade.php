<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ asset('css/navbar.css') }}" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
</head>

<nav class="navbar">
    <div class="nav-wrapper">
        <div class="nav2">
            <img src="{{ asset('elements/logo.png') }}" class="logo">
            <nav class="nav3">
                <ul>
                    <li><a href="{{ route('dashboard.page') }}">Dashboard</a></li>
                    <li><a href="{{ route('community.page') }}">Community</a></li>
                    <li><a href="">Plans</a></li>
                </ul>
            </nav>
        </div>
        <div class="profile">
            <img src="{{ asset('elements/Profile.png') }}" alt="">
            <div class="profile-content">
                @if(isset($user))
                <div class="name"><a href="{{ route('profile.page') }}">{{ $user->full_name }}</a></div>
                <div class="level"> Lvl. 12</div>
                @endif
            </div>
        </div>
    </div>
</nav>


</html>