<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Task extends Model
{
    use HasFactory;

    protected $primaryKey = 'task_id';
    protected $table = 'task';
    protected $fillable = [
        'task_id',
        'lesson_id',
        'task_name',
        'task_description',
        'task_content'
    ];
}
