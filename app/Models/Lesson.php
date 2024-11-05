<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Lesson extends Model
{
    use HasFactory;

    protected $primaryKey = 'lesson_id';
    protected $table = 'lesson';
    protected $fillable = [
        'lesson_id',
        'lesson_name',
        'lesson_desc',
    ];
}
