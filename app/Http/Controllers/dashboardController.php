<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Lesson;
use App\Models\Progress;
use App\Models\User;
use Illuminate\Support\Facades\Session;
use App\Models\Task;

class dashboardController extends Controller
{

    public function showDashboard(Request $request)
    {
        $userID = $request->session()->get('user_id');
        $user = User::find($userID);

        // Fetch all lessons from the database
        $lessons = Lesson::all();

        // Pass the user and lessons data to the dashboard view
        return view('dashboard', compact('user', 'lessons'));
    }

    public function showLessonTasks($lesson_id)
    {
        $user_id = session('user_id'); 

        $tasks = Task::where('lesson_id', $lesson_id)->get();
        $currentTask = $tasks->first();

        $progress = Progress::where('lesson_id', $lesson_id)
        ->where('user_id', $user_id)
        ->first();

        if ($currentTask && $currentTask->task_content) {
            $currentTask->task_content = base64_encode($currentTask->task_content);
        }

        return view('task', compact('tasks', 'currentTask','progress'));
    }

    public function showTaskContent($lesson_id, $taskID)
    {
        $user_id = session('user_id'); 

        $tasks = Task::where('lesson_id', $lesson_id)->get();
        $currentTask = Task::find($taskID);

        $progress = Progress::where('lesson_id', $lesson_id)
        ->where('user_id', $user_id)
        ->first();

        if ($currentTask && $currentTask->task_content) {
            $currentTask->task_content = base64_encode($currentTask->task_content);
        }

        return view('task', compact('tasks', 'currentTask','progress'));
    }
}
