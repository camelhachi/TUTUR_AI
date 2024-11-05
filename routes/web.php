<?php

use App\Http\Controllers\dashboardController;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\Facades\View;
use App\Http\Controllers\userController;
use App\Http\Controllers\TaskController;
use SebastianBergmann\CodeCoverage\Report\Html\Dashboard;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "web" middleware group. Make something great!
|
*/

Route::get('/', function () {
    return view('index');
})->name('index.page');

Route::get('/signup', function () {
    return view('signup');
})->name('signup.page');

Route::get('/login', function () {
    return view('login');
})->name('login.page');

Route::get('/dashboard', function () {
    return view('dashboard');
})->name('dashboard.page');

Route::get('/community', function () {
    return view('community');
})->name('community.page');

Route::get('/profile', function () {
    return view('profile');
})->name('profile.page');

Route::get('/packageDetail', function () {
    return view('packageDetail');
})->name('packageDetail.page');

Route::get('/communityDetail', function () {
    return view('communityDetail');
})->name('communityDetail.page');

Route::post('/createUser', [userController::class, 'createUser'])->name('create.user');


// Route::get('/dashboard', [dashboardController::class, 'showDashboard'])->name('dashboard.page');
Route::get('/dashboard', [dashboardController::class, 'showDashboard'])->name('dashboard.page');

Route::get('/profile', [UserController::class, 'showProfileDetails'])->name('profile.page');


Route::match(['get', 'post'], '/loginUser', [userController::class, 'loginUser'])->name('login.user');


Route::post('/logout', [UserController::class, 'logoutUser'])->name('logout.user');

Route::post('/editProfile', [UserController::class, 'updateProfile'])->name('update.profile');

Route::get('/lessons/{lesson_id}', [dashboardController::class, 'showLessonTasks'])->name('task.page');

Route::get('/lessons/{lesson_id}/tasks/{task_id}', [dashboardController::class, 'showTaskContent'])->name('task.page2');

Route::get('/assesment-view/{lesson_id}/{progress_id}', [taskController::class, 'showAssesment'])->name('assesment.page');

Route::post('/complete-progress/{lesson_id}/{progress_id}', [taskController::class, 'completeProgress'])->name('completeProgress');




View::composer('navbar', function ($view) {
    //Retrieve & share profile data
    $userID = session('user_id');
    $user = \App\Models\User::where('user_id', $userID)->first();

    $view->with('user', $user);
});
