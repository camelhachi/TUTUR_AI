<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ asset('css/profile.css') }}" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
    <title>Profile</title>
</head>

<body>
    <!-- Navigation -->
    @include('navbar');
    <!-- content -->
    <div class="content">
        <div class="container">
            <div class="profile-container">
                <h2>Profile</h2>
                <div class="pic"></div>
                <div class="field">
                    <div class="username">
                        <div class="title">Username</div>
                        <div class="username-container">
                            <div class="text">{{ $user->username }}</div>
                        </div>
                    </div>
                    <div class="edit-form">
                        <form action="{{ route('update.profile') }}" method="POST">
                            @csrf
                            <label for="name">Name</label>
                            <input type="text" id="name" name="name" placeholder="{{ $user->full_name }}">
                            <label for="email">Email</label>
                            <input type="text" id="email" name="email" placeholder="{{ $user->email }}">
                            <button class="update" type="submit">Update <span class="material-symbols-outlined">
                                    trending_flat
                                </span></button>
                        </form>
                        <a href="">Delete your account</a>
                    </div>
                </div>
                <form action="{{ route('logout.user') }}" method="POST">
                    @csrf
                    <button class="logoutbutton" type="submit" class="btn btn-link">Logout <span class="material-symbols-outlined">
                            logout
                        </span></button>
                </form>
            </div>

        </div>
    </div>
</body>